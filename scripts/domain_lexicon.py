#!/usr/bin/env python3
"""Elicit the target domain's own vocabulary, from the domain brief alone.

    python3 scripts/domain_lexicon.py --brief specs/domain_brief_3item.txt --lang it
    python3 scripts/domain_lexicon.py --brief ... --lang en --n 5 --model claude-opus-5

Written for `build_embargo.py`'s middle band.  Frequency tells us whether a word
is rare in the language; it cannot tell us whether a rare word is the author's
signature or a specialist term that any text about the target domain might reach
for.  `terrapieni` and `promontorio` sit within 0.7 zipf of each other and belong
on opposite sides of that line.

The discriminator is contrast, and the cheapest contrast available is the target
domain itself: ask a model, given ONLY the brief, what words a description of this
domain would naturally use.  A word on that list is the domain's property, whoever
else used it.  The model never sees the source, so nothing about the source can
leak into the decision -- which is also what makes the step generic: the input is
the brief the experiment already has, for any author and any domain.

Sampled n times at temperature 1 and kept by vote count, because a single sample
is a lucky draw rather than a lexicon.
"""
import argparse
import json
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request

D = pathlib.Path(__file__).resolve().parent.parent
SPEC = D / "spec"
ENV = D.parent / ".env"

LANG = {"it": "Italian", "en": "English", "fr": "French"}

PROMPT = """You are compiling a vocabulary list for a writing experiment.

DOMAIN:
{brief}

List single words in {language} that a purely DESCRIPTIVE passage about this domain \
would naturally use: concrete nouns for places, terrain, water, structures and \
equipment; verbs of position, extent, movement and change; adjectives of shape, size, \
texture and light; adverbs of place and manner.

Rules: lemmas only (singular nouns, infinitive verbs, masculine singular adjectives). \
No proper names. No phrases. No numbering, no commentary, no blank lines. \
One word per line, about 200 words."""

# Eliciting a list has a recall ceiling: five samples of ~280 words covered only 6
# of the 38 words we needed a verdict on, so absence from the list meant nothing
# and the signal could only ever argue for freeing.  Classification has no such
# ceiling -- every word put to it comes back with an answer.  The model still never
# sees the source, so its verdict is about the domain and nothing else.
CLASSIFY = """You are helping decide which words belong to a domain's own vocabulary.

DOMAIN:
{brief}

For each word below, decide whether a purely DESCRIPTIVE passage about this domain could naturally use it — the topography, the water, the structures, the light, the institutional presence, and the ordinary verbs and adjectives such description needs.

Answer YES if the word is at home in a description of this domain.
Answer NO if it belongs to some other world — another landscape, another economy, another period, another register — and would read as imported.

Judge the word itself, not any particular sentence. Output exactly one line per word, in the given order, formatted `word: YES` or `word: NO`. No commentary.

WORDS:
{words}"""

# The domain judge answers "is this word at home in the target domain?", which is
# the wrong question for `ameno`, `considerabile`, `variato`: those are not foreign
# to a Mediterranean description, they are foreign to present-day prose.  Register
# is a second, orthogonal axis, and it needs its own question -- asked about the
# language, with no mention of the source or its period, so the step stays generic.
# First version of this asked whether a word reads as "literary, elevated, dated or
# archaic", which conflates four things and produced the wrong answers: `falde`,
# `alture`, `lembo`, `volge` came back MARKED, and they are indeed literary — but
# the experiment asks for MODERN LITERARY prose, so literary is not a reason to
# forbid anything.  The question that matters is period, and elevation has to be
# explicitly allowed.
PERIOD = """You are sorting words by period in {language}.

For each word below, decide where it sits in the history of the language.

Answer CURRENT if a writer of literary prose today could use it unselfconsciously. \
Elevated, poetic and technical vocabulary counts as CURRENT — the question is not \
whether the word is ordinary but whether it is still in use.
Answer DATED if it belongs to an earlier stage of {language} — nineteenth-century or \
older — such that a writer today would reach for a different word, and a reader would \
place it in the past.

Judge the word itself, out of context. Output exactly one line per word, in the given \
order, formatted `word: CURRENT` or `word: DATED`. No commentary.

WORDS:
{words}"""


def env(k):
    for line in ENV.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(k + "="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit(f"domain_lexicon: {k} not found in {ENV}")


def ask(model, prompt, max_tokens):
    # Opus 5 thinks by default and will spend the whole budget in the thinking
    # block, returning no text block at all -- which is what happened on the first
    # run here (one sample of 270 words, then four empty ones).  A word list needs
    # no deliberation, so thinking is off and temperature does the sampling.
    body = {"model": model, "max_tokens": max_tokens, "temperature": 1,
            "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
        headers={"x-api-key": env("ANTHROPIC_API_KEY"),
                 "anthropic-version": "2023-06-01",
                 "content-type": "application/json"}, method="POST")
    for attempt in range(5):
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=300).read())
            txt = "".join(b.get("text", "") for b in r["content"]
                          if b.get("type") == "text")
            stop = r.get("stop_reason")
            if stop not in (None, "end_turn", "stop_sequence"):
                # `refusal` truncates the answer mid-list and looks exactly like a
                # token limit: a word list holding `fanciulle` and `modestia` next
                # to each other reads as something else to the classifier.  Small
                # chunks are the workaround; report it either way.
                print(f"  stop_reason={stop} after "
                      f"{r.get('usage', {}).get('output_tokens')} tokens")
            return txt
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529) and attempt < 4:
                time.sleep(min(3 * 2 ** attempt, 40))
                continue
            sys.exit(f"domain_lexicon: HTTP {e.code} {e.read()[:200]}")
        except (TimeoutError, urllib.error.URLError, OSError) as e:
            # a hung read must not lose the samples already collected
            if attempt < 4:
                print(f"  retrying after {type(e).__name__}")
                time.sleep(min(3 * 2 ** attempt, 40))
                continue
            print(f"  giving up on this sample: {type(e).__name__}")
            return ""


def words(text, lang):
    out = []
    for line in text.splitlines():
        w = line.strip().strip("-*•·").strip().lower()
        if re.fullmatch(r"[a-zà-ÿ'’-]{2,}", w):
            out.append(w.replace("’", "'"))
    return out


# per question: the template, the positive answer, and the pair of labels
QUESTIONS = {
    "domain": (CLASSIFY, "yes", r"(yes|no)"),
    "period": (PERIOD, "current", r"(current|dated)"),
}


def verdicts(text, wanted, alt):
    """parse `word: ANSWER` lines, keeping only words we asked about."""
    out = {}
    for line in text.splitlines():
        m = re.match(rf"\s*[-*]?\s*([a-zà-ÿ'’-]+)\s*[:\-—]\s*{alt}\b", line, re.I)
        if m and m.group(1).lower() in wanted:
            out[m.group(1).lower()] = m.group(2).lower()
    return out


def classify(a, brief):
    """put the filter's decide band to a source-blind judge, n times."""
    src = json.loads(pathlib.Path(a.classify).read_text(encoding="utf-8"))
    band = src["decide"] if isinstance(src, dict) else list(src)
    if not band:
        sys.exit("domain_lexicon: nothing in the decide band")
    template, positive, alt = QUESTIONS[a.question]
    wanted = {w.lower() for w in band}
    yes = {w.lower(): 0 for w in band}
    seen = {w.lower(): 0 for w in band}
    chunks = [band[i:i + a.chunk] for i in range(0, len(band), a.chunk)]
    for i in range(a.n):
        answered = 0
        for c in chunks:
            prompt = template.format(brief=brief, language=LANG[a.lang],
                                     words="\n".join(c))
            got = verdicts(ask(a.model, prompt, a.max_tokens), wanted, alt)
            for w, v in got.items():
                seen[w] += 1
                yes[w] += (v == positive)
            answered += len(got)
        print(f"  sample {i + 1}: {answered}/{len(band)} words answered "
              f"in {len(chunks)} chunks")
    out = (pathlib.Path(a.out) if a.out else
           SPEC / f"{a.question}_verdicts_{a.lang}.json")
    # MERGE, never replace: judging one newly-added band word used to overwrite the
    # file and silently discard the verdicts for every other word in it.
    kept = {}
    if out.exists():
        prev = json.loads(out.read_text(encoding="utf-8")).get("verdicts", {})
        kept = {w: d for w, d in prev.items() if w.lower() not in yes}
    out.write_text(json.dumps({
        "lang": a.lang, "question": a.question, "model": a.model, "n": a.n,
        "brief": brief if a.question == "domain" else None,
        "positive": positive,
        "note": "source-blind: the judge saw the question and the word list only",
        "verdicts": dict(sorted(
            ({w: {"yes": yes[w.lower()], "answered": seen[w.lower()]} for w in band}
             | kept).items(), key=lambda kv: (-kv[1]["yes"], kv[0]))),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    unan = sum(1 for w in yes if seen[w] and yes[w] in (0, seen[w]))
    print(f"wrote {out.relative_to(D)}: {len(band)} words, "
          f"{unan} unanimous, {len(band) - unan} split")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brief", required=True)
    ap.add_argument("--classify", metavar="FILTERED.json",
                    help="classify the decide band of a build_embargo --write file "
                         "instead of eliciting a lexicon")
    ap.add_argument("--question", default="domain", choices=sorted(QUESTIONS),
                    help="domain: is this thing at home in the target domain? "
                         "(ask of nouns) "
                         "period: is it still in use, or from an earlier stage "
                         "of the language? (ask of all) "
                         "The two are orthogonal and the band needs both")
    ap.add_argument("--lang", default="it", choices=sorted(LANG))
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--n", type=int, default=5, help="samples to pool (default 5)")
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--chunk", type=int, default=12,
                    help="words per classification call (default 12): a refusal "
                         "stop then costs one chunk instead of the rest of the list")
    ap.add_argument("--out", help="default spec/domain_lexicon_<lang>.json")
    a = ap.parse_args()

    brief = pathlib.Path(a.brief).read_text(encoding="utf-8").strip()
    if a.classify:
        return classify(a, brief)
    prompt = PROMPT.format(brief=brief, language=LANG[a.lang])
    votes, samples = {}, []
    for i in range(a.n):
        got = words(ask(a.model, prompt, a.max_tokens), a.lang)
        samples.append(len(got))
        for w in set(got):
            votes[w] = votes.get(w, 0) + 1
        print(f"  sample {i + 1}: {len(got)} words, pool now {len(votes)}")

    out = pathlib.Path(a.out) if a.out else SPEC / f"domain_lexicon_{a.lang}.json"
    out.write_text(json.dumps({
        "lang": a.lang, "brief": brief, "model": a.model, "n": a.n,
        "sample_sizes": samples,
        "note": "elicited from the brief alone; the model never saw the source text",
        "votes": dict(sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    hist = {v: sum(1 for x in votes.values() if x == v) for v in range(1, a.n + 1)}
    print(f"wrote {out.relative_to(D)}: {len(votes)} types; by votes {hist}")


if __name__ == "__main__":
    main()
