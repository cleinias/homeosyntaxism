#!/usr/bin/env python3
"""Filter the source-word embargo down to authentic source markers.

    python3 scripts/build_embargo.py --lang it
    python3 scripts/build_embargo.py --lang it --review > review.md

`build_specs.py` builds the embargo as *every* content word of the source minus a
three-word whitelist.  That is a hammer for a mosquito: on the Italian incipit it
forbids 284 forms, of which 104 are very common Italian (`tutto`, `parte`,
`tempo`, `giorno`, `città`) and another band is ordinary landscape and maritime
vocabulary that the source and a Mediterranean scene legitimately share.  The
embargo's job is to prevent *calque* — reuse of what is distinctively the
source's — so what it should contain is the source's markers, not its language.

Five layers, applied in order; every verdict carries the layer that produced it:

  NAME      proper names and the adjectives of place and nation that identify the
            source -> always embargoed, whatever their frequency ('Como' is a
            common string).
  ARCHAIC   apocope, elision, prosthetic i-, archaic adverbs -> freed, because
            the prompt already says "Write MODERN literary Italian (no apocope,
            no archaic elisions)".  Freeing them costs nothing and removes the
            largest block of noise.
  COMMON    frequency at or above --zipf-free -> freed as common property.
  MARKER    frequency below --zipf-mark -> embargoed as a marker.
  DECIDE    the band between -> left for a human, with one corroborating signal:
            whether the word turns up in the source-blind domain generations we
            already have.  Those runs had the old embargo in force, so a word
            appearing there anyway is good evidence it is natural to the target
            domain; absence proves nothing, which is why this only ever argues
            for freeing.

Frequencies come from `wordfreq` when it is importable and otherwise from the
cached table in spec/, which this script writes.  The cache is what makes the
classification reproducible without the package.
"""
import argparse
import json
import pathlib
import re
import sys
import unicodedata

D = pathlib.Path(__file__).resolve().parent.parent
SPEC = D / "spec"

# Source-identifying names, and the place/nation adjectives that work like names.
# Frequency cannot decide these: 'como' and 'san' are common strings.
NAMES = {
    "it": {"como", "adda", "lecco", "resegone", "milano", "san", "martino",
           "lombarda", "spagnoli"},
    "en": {"como", "adda", "lecco", "resegone", "milan", "martin", "martino",
           "lombard", "spanish"},
}

# Layer 2, Italian: forms the modernity clause already rules out.
ARCHAIC_IT_RE = [
    r"[aeio]van$|[aeio]an$",   # apocopated 3pl: insegnavan, accarezzavan, lascian
    r"[aei]r$",                # apocopated infinitive: diradar, prender, finir
    r"^is(?=[bcdfgpqtv])",     # prosthetic i-: iscoprite, ispianate.  NOT ^in-,
]                              # which would swallow intorno, interrotte, incammina
# Apocope, elision and forms whose *spelling* is archaic (`ristringersi` for
# restringersi, `pendìo` for pendio).  Modern connectives that merely feel
# literary — perciò, purché, anzi — are deliberately NOT here: frequency frees
# them, and the label should say why they were freed.
ARCHAIC_IT = {"que", "coll", "anch", "dall", "sol", "men", "ancor", "par",
              "vien", "pur", "gran", "buon", "talché", "talche", "tosto",
              "ivi", "donde", "ristringersi", "pendìo", "pendio"}
NOT_ARCHAIC_IT = {"corso", "lento", "ameno", "vero", "onore", "nome", "fronte",
                  "sale", "orna", "estate", "vista", "voce", "forma", "intorno",
                  "interrotte", "incammina", "ingrossa", "rincomincia",
                  "discerna", "alleggerire", "somigliare", "considerabile"}


# Enclitic pronouns: the source writes `distendersi`, `perdendosi`, `rallentarsi`,
# and Stanza keeps the clitic in the lemma, so a frequency lookup on the lemma
# alone reports an ordinary verb as rare (`perdendosi` 2.50 vs `perdendo` 4.36).
CLITICS = ("si", "mi", "ti", "ci", "vi", "ne", "lo", "la", "li", "le", "gli")


def variants(form, lemma):
    """every string worth a frequency lookup for one embargo entry."""
    out = {form, lemma}
    for w in (form, lemma):
        for c in CLITICS:
            if w.endswith(c) and len(w) > len(c) + 3:
                stem = w[:-len(c)]
                out |= {stem, stem + "e"}      # distender -> distendere
    return {v for v in out if v}


def fold(s):
    return "".join(c for c in unicodedata.normalize("NFD", s.lower())
                   if unicodedata.category(c) != "Mn")


def lemma_table(lang, source=None, build=False):
    """form -> (lemma, upos).  Frequency has to be read off the lemma: `golfi`
    scores 2.24 and `golfo` 4.19, and the Italian embargo is a list of surface
    forms, so scoring forms alone marks ordinary words as rare.

    --build-lemma-cache parses the source with Stanza (run it where Stanza is
    installed, e.g. the Atlas venv) and writes the cache the normal path reads.
    """
    cache = SPEC / f"lemma_{lang}.csv"
    if build:
        import stanza
        nlp = stanza.Pipeline(lang, processors="tokenize,pos,lemma",
                              download_method=None, verbose=False)
        t = {}
        for s in nlp(source).sentences:
            for w in s.words:
                # accents kept: wordfreq scores `estremità` 3.62 and `estremita` 0
                f, l = w.text.lower(), (w.lemma or w.text).lower()
                if re.fullmatch(r"[a-zà-ÿ']+", f) and re.fullmatch(r"[a-zà-ÿ']+", l):
                    t.setdefault(f, (l, w.upos))
        cache.write_text("form,lemma,upos\n" + "".join(
            f"{f},{l},{p}\n" for f, (l, p) in sorted(t.items())), encoding="utf-8")
        print(f"wrote spec/{cache.name}: {len(t)} forms")
        return t
    if not cache.exists():
        # The English embargo is built from lemmas already (build_specs.py), so
        # identity is correct there; for Italian it is a list of surface forms and
        # the cache is not optional.
        if lang == "en":
            print(f"note: no spec/{cache.name}; treating entries as lemmas",
                  file=sys.stderr)
            return {}
        sys.exit(f"build_embargo: no spec/{cache.name}; run once with "
                 f"--build-lemma-cache where Stanza is installed")
    t = {}
    for line in cache.read_text(encoding="utf-8").splitlines()[1:]:
        f, l, p = line.split(",", 2)
        t[f] = (l, p)
    return t


def zipf_table(words, lang):
    """word -> zipf, from wordfreq if available, else from the cached csv."""
    cache = SPEC / f"zipf_{lang}.csv"
    try:
        from wordfreq import zipf_frequency
        t = {w: round(zipf_frequency(w, lang), 2) for w in words}
        cache.write_text("word,zipf\n" + "".join(f"{w},{v}\n" for w, v in
                                                 sorted(t.items())), encoding="utf-8")
        return t, "wordfreq"
    except ImportError:
        if not cache.exists():
            sys.exit(f"build_embargo: no wordfreq and no cache at {cache.name}")
        t = {}
        for line in cache.read_text(encoding="utf-8").splitlines()[1:]:
            w, v = line.rsplit(",", 1)
            t[w] = float(v)
        missing = [w for w in words if w not in t]
        if missing:
            sys.exit(f"build_embargo: {len(missing)} words absent from the cached "
                     f"table (e.g. {missing[:5]}); re-run where wordfreq is installed")
        return t, f"cache {cache.name}"


# The English prompt likewise says "Write MODERN literary English", so the same
# reasoning frees Colquhoun's archaisms.
ARCHAIC_EN = {"thence", "whence", "hither", "thither", "betwixt", "whither",
              "thereupon", "ere", "nigh"}


def is_archaic(w, lang):
    if lang == "en":
        return w in ARCHAIC_EN
    if lang != "it":
        return False
    if w in ARCHAIC_IT:
        return True
    if w in NOT_ARCHAIC_IT:
        return False
    return any(re.search(rx, w) for rx in ARCHAIC_IT_RE)


def domain_words(lang):
    """content words of the source-blind three-item-brief generations we have."""
    # Globbed rather than named: the corroboration is "whatever source-blind runs in
    # this domain we have", and naming them pinned it to the dom3 files, which were
    # generated against the superseded source and are now in legacy-pre-1840/.
    runs = sorted(
        [q for q in (D / "results").glob(f"unrestricted_{lang}_*.json")
         if "legacy" not in q.parts]
        + [q for q in (D.parent / "reproducibility/graded-corpus/results").glob(
            f"unrestricted_{lang}_*.json") if "legacy" not in q.parts])
    out, seen = set(), []
    for p in runs:
        if not p.exists():
            continue
        seen.append(p.name)
        d = json.loads(p.read_text(encoding="utf-8"))
        for g in d.get("gens", []):
            t = g if isinstance(g, str) else g.get("text", "")
            out |= set(re.findall(r"[a-zà-ÿ']{3,}", fold(t)))
    return out, seen


def rulings(lang):
    """Human decisions on the band, applied last and overriding everything.

    A judge that is unanimous can still be wrong -- `degradando` came back DATED
    three times out of three and is ordinary modern Italian -- so the critic's
    ruling is the final layer, and it lives in spec/ as data rather than in this
    file as a special case.
    """
    path = SPEC / f"embargo_rulings_{lang}.json"
    if not path.exists():
        return {}
    d = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for verdict, key in (("free", "free"), ("EMBARGO", "embargo")):
        for w, why in (d.get(key) or {}).items():
            out[w.lower()] = (verdict, why)
    return out


def judged(path):
    """word -> (yes, answered) from a domain_lexicon.py verdicts file, if present."""
    if not path.exists():
        return {}
    v = json.loads(path.read_text(encoding="utf-8"))["verdicts"]
    return {w.lower(): (d["yes"], d["answered"]) for w, d in v.items()}


def band_call(k, upos, period, domain, in_gens):
    """Resolve one middle-band word from the two source-blind judges.

    Two orthogonal questions, and which one applies depends on the part of speech:
    *period* (is the word still in use?) is asked of everything, because a dated
    word is a source marker whatever it names; *domain* (is this thing at home
    here?) is only well posed for nouns — asking it of `somigliare` or `spaziare`
    returns NO for a verb that is simply generic, not foreign.

    A human is called only where the samples split, or where a judge contradicts
    the one piece of observed evidence we have: the word turning up in the
    source-blind generations already run.
    """
    pv, pn = period.get(k, (0, 0))
    dv, dn = domain.get(k, (0, 0))
    noun = upos in ("NOUN", "PROPN")
    if not pn:
        return "decide", "no period verdict"
    if 0 < pv < pn or (noun and dn and 0 < dv < dn):
        return "decide", f"judges split (period {pv}/{pn}, domain {dv}/{dn})"
    if pv * 2 < pn:
        if in_gens:
            return "decide", "judged dated, but the domain generations use it"
        return "EMBARGO", f"dated ({pv}/{pn} current)"
    if noun and dn and dv * 2 < dn:
        if in_gens:
            return "decide", "judged foreign, but the domain generations use it"
        return "EMBARGO", f"names something foreign to the domain ({dv}/{dn})"
    return "free", f"current{' and at home' if noun else ', not a noun'}"


def classify(words, lang, zipf, free, mark, dom, lemmas, archaisms="free",
             period=None, domain=None, ruled=None):
    period, domain, ruled = period or {}, domain or {}, ruled or {}
    for w in words:
        k = w.lower()
        f = fold(k)
        lem, upos = lemmas.get(k, lemmas.get(f, (k, "")))
        if k in ruled:
            verdict, why = ruled[k]
            yield w, max(zipf.get(x, 0.0) for x in variants(k, lem)), verdict, "RULING", why
            continue
        # a word is as common as its most common form: `golfi` 2.24 / `golfo` 4.19,
        # `perdendosi` 2.50 / `perdendo` 4.36
        v = max(zipf.get(x, 0.0) for x in variants(k, lem))
        if f in NAMES.get(lang, ()) or upos == "PROPN":
            yield w, v, "EMBARGO", "NAME", f"source-identifying ({upos or 'listed'})"
        elif is_archaic(f, lang):
            if archaisms == "free":
                yield w, v, "free", "ARCHAIC", "the register clause already rules it out"
            else:
                yield w, v, "EMBARGO", "ARCHAIC", "archaisms embargoed by request"
        elif v >= free:
            yield w, v, "free", "COMMON", f"zipf {v:.2f} (lemma {lem}) >= {free}"
        elif v < mark:
            yield w, v, "EMBARGO", "MARKER", f"zipf {v:.2f} (lemma {lem}) < {mark}"
        else:
            verdict, why = band_call(k, upos, period, domain, f in dom)
            if verdict == "decide" and not period:
                why = ("appears in the source-blind domain generations" if f in dom
                       else "not seen in the domain generations")
            yield w, v, verdict, "BAND" if period else "DECIDE", why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="it", choices=["it", "en"])
    ap.add_argument("--degree", default="D2")
    ap.add_argument("--zipf-free", type=float, default=3.5,
                    help="at or above this, a word is common property (default 3.5)")
    ap.add_argument("--zipf-mark", type=float, default=2.5,
                    help="below this, a word is a source marker (default 2.5)")
    ap.add_argument("--review", action="store_true", help="print the review tables")
    ap.add_argument("--write", action="store_true",
                    help="write spec/embargo_<lang>_filtered.json")
    ap.add_argument("--archaisms", default="free", choices=["free", "embargo"],
                    help="what to do with apocope and archaic spellings. Default free, "
                         "because the prompt's register clause ('MODERN literary "
                         "Italian') already forbids them; pass embargo when the target "
                         "domain is itself period — e.g. a Calvino text transposed into "
                         "an eighteenth-century world, where archaism is the easy calque")
    ap.add_argument("--build-lemma-cache", action="store_true",
                    help="parse the source with Stanza and write spec/lemma_<lang>.csv")
    ap.add_argument("--from-prompt", metavar="RUN.json",
                    help="take the embargo (and source) from a recorded production "
                         "run instead of a spec file — the only place the Manzoni "
                         "English embargo exists")
    a = ap.parse_args()

    if a.from_prompt:
        spec_path = pathlib.Path(a.from_prompt)
        prompt = json.loads(spec_path.read_text(encoding="utf-8"))["prompt"]
        body = prompt.split("Use different words.", 1)[1].split("REQUIREMENTS")[0]
        emb = sorted({w.strip() for w in body.replace("\n", " ").split(",")
                      if w.strip() and " " not in w.strip()})
        src = prompt.split("«", 1)[1].split("»", 1)[0] if "«" in prompt else ""
        spec = {"embargo": emb, "source": src}
    else:
        spec_path = SPEC / f"specs_{a.lang}_{a.degree}.json"
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    emb = spec["embargo"]
    if a.build_lemma_cache:
        sys.path.insert(0, str(D / "scripts"))
        import run_production as RP
        lemma_table(a.lang, RP.resolve_source(spec, D), build=True)
        return
    lemmas = lemma_table(a.lang)
    lookups = set()
    for w in emb:
        k = w.lower()
        lookups |= variants(k, lemmas.get(k, lemmas.get(fold(k), (k, "")))[0])
    zipf, src = zipf_table(sorted(lookups), a.lang)
    dom, dom_files = domain_words(a.lang)
    period = judged(SPEC / f"period_verdicts_{a.lang}.json")
    domain = judged(SPEC / f"domain_verdicts_{a.lang}.json")
    ruled = rulings(a.lang)
    rows = list(classify(emb, a.lang, zipf, a.zipf_free, a.zipf_mark, dom, lemmas,
                         a.archaisms, period, domain, ruled))

    keep = [r for r in rows if r[2] == "EMBARGO"]
    free = [r for r in rows if r[2] == "free"]
    dec = [r for r in rows if r[2] == "decide"]
    print(f"# Embargo filter — {a.lang}, from spec/{spec_path.name}\n")
    print(f"frequencies: {src} · domain corroboration from: "
          f"{', '.join(dom_files) or 'none found'} ({len(dom)} word types)\n")
    print(f"- **{len(emb)} forms in** → **{len(keep)} embargoed**, {len(free)} freed, "
          f"{len(dec)} to decide")
    if period or domain:
        print(f"  band judges: period {len(period)} words, domain {len(domain)} words")
    if ruled:
        print(f"  human rulings applied: {len(ruled)}")
    for layer in ["NAME", "MARKER", "ARCHAIC", "COMMON", "BAND", "RULING"]:
        n = len([r for r in rows if r[3] == layer])
        print(f"  - {layer}: {n}")

    if a.review:
        for title, rs in [("Embargoed — names", [r for r in keep if r[3] == "NAME"]),
                          ("Embargoed — markers", [r for r in keep if r[3] == "MARKER"]),
                          ("To decide", dec),
                          ("Freed — archaic/apocope", [r for r in free if r[3] == "ARCHAIC"]),
                          ("Freed — common", [r for r in free if r[3] == "COMMON"])]:
            print(f"\n## {title} ({len(rs)})\n")
            if title == "To decide":
                print("| word | zipf | corroboration |")
                print("|---|---:|---|")
                for w, v, _, _, why in sorted(rs, key=lambda r: -r[1]):
                    print(f"| `{w}` | {v:.2f} | {why} |")
            else:
                print(", ".join(f"`{w}`" for w, *_ in sorted(rs, key=lambda r: -r[1])))

    if a.write:
        out = SPEC / f"embargo_{a.lang}_filtered.json"
        out.write_text(json.dumps({
            "lang": a.lang, "from": spec_path.name,
            "thresholds": {"zipf_free": a.zipf_free, "zipf_mark": a.zipf_mark},
            "frequency_source": src,
            "embargo": sorted(w for w, _, v, *_ in rows if v == "EMBARGO"),
            "decide": sorted(w for w, _, v, *_ in rows if v == "decide"),
            "freed": {layer: sorted(w for w, _, v, l, _ in rows
                                    if v == "free" and l == layer)
                      for layer in ["ARCHAIC", "COMMON"]},
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nwrote spec/{out.name}")


if __name__ == "__main__":
    main()
