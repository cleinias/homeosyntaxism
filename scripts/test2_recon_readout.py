#!/usr/bin/env python3
"""test2_recon_readout.py — test 2 (the per-pattern probe), original and revised wordings, for a hand check.

SF, 2026-11-27. Test 2 sent each pattern entry alone, in its original (July) and its revised wording,
in two prompts: identity (from which work, by which author, was this specification derived?) and
reconstruction (what did the source passage contain, who acted in it, which of its words does the
specification echo?), five times per model. This script lays out both prompts verbatim, the 1840
incipit in full, a summary table, and, entry by entry, the specification as sent, the works named
under the identity prompt, and every reconstruction answer.

The reconstruction count is that of the answers that reconstruct Manzoni's passage, read by hand
(SF: the question is whether Manzoni's text was reconstructed, not whether some text was; a first
version counted answers containing a word such as "river" and is dropped). The hand reading, with its
criterion and a note per item, is results/test2_recon_hand_reading_1127.json, keyed by item id. An
answer whose AUTHOR or TITLE names Manzoni or the novel counts in any case.

Refusals: run_probe_batch.py stopped an item at its first refusal, so a refused item has one refusal
and the tries after it were never sent. The table gives the refusals per cell; the caption says so.

Read-only. Writes the markdown to --out (default results/test2_recon_readout.md).

  python3 scripts/test2_recon_readout.py --out "../../../Docs and Research notes/PoeticsToday/..."
"""
import argparse, json, re
from collections import Counter
from pathlib import Path

HS = Path(__file__).resolve().parent.parent
RES = HS / "results"
MANIFEST = HS / "specs" / "probe" / "loopA_patterns_it_D2.json"
INCIPIT = HS / "texts" / "incipit_it.txt"
HAND = RES / "test2_recon_hand_reading_nopre_1127.json"
# Opus 5: the 11-27 rerun (frame without its opening line; refusals recorded, every try sent).
# Qwen3-235B: the 11-26 run, original frame (it never refused). OPUS_1126 is shown as a check.
MANIFEST_NOPRE = HS / "specs" / "probe" / "loopA_patterns_it_D2_nopre.json"
MODELS = {"Opus 5": "probe_loopA_patterns_it_D2_nopre_loopA-1127_claude-opus-5.json",
          "Qwen3-235B": "probe_loopA_patterns_it_D2_loopA-1126_Qwen_Qwen3-235B-A22B-Instruct-2507-FP8.json"}
OPUS_1126 = "probe_loopA_patterns_it_D2_loopA-1126_claude-opus-5.json"
N = 5  # tries per item
FIELDS = [("content", "Content"), ("agents", "Agents"), ("echoed_words", "Echoed words")]
WORDING = {"july": "original (July) wording", "revised": "revised wording", "both": "wording unchanged"}
CJK = re.compile(r"[　-鿿豈-﫿]+")


def esc(s):
    # the PDF font has no CJK glyphs: a Chinese run inside an answer (a Qwen habit) is marked, not printed
    s = CJK.sub(lambda m: f"[{len(m.group(0))} Chinese characters in the original]", s)
    s = s.replace("\\", "\\\\").replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(r"(?<!\*)\*(?!\*)", r"\\*", s)   # lone asterisks from the answers


def quote(text):
    """A prompt as a quoted line block: line breaks kept, long lines wrapped."""
    return "\n".join("> |" + (" " + esc(l) if l else "") for l in text.split("\n"))


def spec_of(it):
    return it["prompt"].split("SPECIFICATION:\n", 1)[1].split("\n\nAnswer in exactly", 1)[0]


def frame(it):
    """The prompt as sent, the pattern entry replaced by a placeholder (checked to occur once)."""
    spec = spec_of(it)
    assert it["prompt"].count(spec) == 1
    return it["prompt"].replace(spec, "[ENTRY]").rstrip()


def tally(r):
    """answers given, refusals, tries not sent, for one results item."""
    # Anthropic: one request per try, a refusal is a response with stop_reason "refusal";
    # vLLM (Qwen): one request with n=5, and it never refused
    refused = sum(1 for x in r["responses"] if x.get("stop_reason") == "refusal")
    unsent = N - len(r["answers"]) - refused
    assert unsent >= 0 and (unsent == 0 or r["failure"]), r["id"]
    return len(r["answers"]), refused, unsent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RES / "test2_recon_readout.md"))
    a = ap.parse_args()
    man = json.loads(MANIFEST.read_text())
    nopre = json.loads(MANIFEST_NOPRE.read_text())
    assert [i["id"] for i in nopre["items"]] == [i["id"] for i in man["items"]]
    res = {m: json.loads((RES / fn).read_text())["items"] for m, fn in MODELS.items()}
    old = json.loads((RES / OPUS_1126).read_text())["items"]
    hand = json.loads(HAND.read_text())
    recon = [i for i in man["items"] if i["question"] == "reconstruction"]
    ident = {(i["pattern"], i["wording"]): i for i in man["items"] if i["question"] == "identity"}
    assert set(hand["entries"]) == {i["id"] for i in recon}, "the hand reading must cover every item"
    patterns = list(dict.fromkeys(i["pattern"] for i in recon))
    order = {"july": 0, "both": 1, "revised": 2}
    recon.sort(key=lambda i: (patterns.index(i["pattern"]), order[i["wording"]]))

    cells, body, totals = {}, [], Counter()
    for it in recon:
        pat, w = it["pattern"], it["wording"]
        h = hand["entries"][it["id"]]
        idi = ident[(pat, w)]
        body += [f"## {pat}, {WORDING[w]}", "", "**Specification as sent:**", "",
                 "\n".join("> " + esc(l.strip()) for l in spec_of(it).split("\n")), "",
                 f"**Reconstruction, read by hand:** {esc(h['note'])}", ""]
        named_works = []
        for m in MODELS:
            ri, rr = res[m][idi["id"]], res[m][it["id"]]
            m_id, z_id, u_id = tally(ri)
            q, z_rc, u_rc = tally(rr)
            n = sum(bool(x.get("recognised")) for x in ri["answers"])
            assert set(h["reconstructs"][m]) <= {x["j"] for x in rr["answers"]}, (it["id"], m)
            p = len(set(h["reconstructs"][m]) | {x["j"] for x in rr["answers"] if x.get("recognised")})
            cells[(pat, w, m)] = (n, m_id, p, q, z_id + z_rc, u_id + u_rc)
            totals[(m, "refused")] += z_id + z_rc
            totals[(m, "unsent")] += u_id + u_rc
            works = Counter((x.get("title") or "?") for x in ri["answers"])
            named_works.append(f"{m}: " + ("; ".join(f"*{esc(t)}*" + (f" ({k})" if k > 1 else "")
                                                     for t, k in works.most_common())
                                           if works else "refused, no answer"))
        body += ["**Identity prompt, works named:** " + ". ".join(named_works) + ".", ""]
        for m in MODELS:
            rr = res[m][it["id"]]
            q, z_rc, u_rc = tally(rr)
            note = (f": {q} answers, {z_rc} refused" + (f", {u_rc} not sent" if u_rc else "")) if z_rc else ""
            body += [f"### {m}, reconstruction{note}", ""]
            if not rr["answers"]:
                body += ["Refused; no answer.", ""]
            for x in rr["answers"]:
                body.append(f"{x['j'] + 1}. *{esc(x.get('author') or '?')}, {esc(x.get('title') or '?')}; "
                            f"confidence {x.get('confidence')}*")
                for f, lab in FIELDS:
                    body.append(f"    - {lab}: {esc(x.get(f) or '(missing)')}")
                if x["j"] in h["reconstructs"][m] or x.get("recognised"):
                    body.append("    - **Reconstructs Manzoni's passage.**")
            body.append("")

    def check_line():
        """The 11-26 Opus run (original frame, stopped at refusals) against the rerun, identity prompt."""
        agree, differ = 0, []
        for i in man["items"]:
            if i["question"] != "identity" or not old[i["id"]]["answers"]:
                continue
            a = sum(bool(x.get("recognised")) for x in old[i["id"]]["answers"])
            b = sum(bool(x.get("recognised")) for x in res["Opus 5"][i["id"]]["answers"])
            if (a > 0) == (b > 0):
                agree += 1
            else:
                differ.append(f"{i['pattern']} {i['wording']} ({a} of {len(old[i['id']]['answers'])} "
                              f"then, {b} of 5 now)")
        return ("Check against the 11-26 Opus run (original frame; it stopped each item at its first "
                f"refusal): of the identity items it answered, {agree} agree with the rerun on whether "
                "the novel is named at all" + (", and " + "; ".join(differ) + (" does not." if len(differ) == 1 else " do not.")
                   if differ else "."))

    def cell(pat, w, m):
        n, mm, p, q, z, u = cells[(pat, w, m)]
        return f"{n}/{mm}\\textbar{{}}{p}/{q}({z})" + ("\\textsuperscript{*}" if u else "")

    tex = ["\\begin{center}\\small", "\\begin{tabular}{@{}l cc @{\\hspace{2em}} cc@{}}", "\\toprule",
           "pattern & \\multicolumn{2}{c}{" + list(MODELS)[0] + "} & \\multicolumn{2}{c}{"
           + list(MODELS)[1] + "} \\\\", "\\cmidrule(lr){2-3}\\cmidrule(l){4-5}",
           " & original & revised & original & revised \\\\", "\\midrule"]
    for pat in patterns:
        row = [pat]
        for m in MODELS:
            if (pat, "both", m) in cells:
                row.append("\\multicolumn{2}{c}{" + cell(pat, "both", m) + "}")
            else:
                row += [cell(pat, "july", m), cell(pat, "revised", m)]
        tex.append(" & ".join(row) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{center}"]

    md = ["---", "title: 'Test 2: the original and revised wordings against the incipit'",
          "subtitle: '2026-11-27. For SF’s check. Generated by homeosyntaxism/scripts/test2_recon_readout.py "
          "from the results files and the hand reading'", "header-includes: |", "  \\usepackage{booktabs}",
          "---", "",
          "Each pattern entry was sent alone, in two separate prompts, an identity prompt and a "
          "reconstruction prompt, five times per model. The six entries the revision rewrote were sent "
          "in their original (July) and their revised wording, the four it left alone once. Opus 5 "
          "(thinking disabled) is the rerun of 11-27, every try sent and each refusal recorded, in the "
          "frame below; Qwen3-235B (ACES, 11-26) had the same frame with one more opening line, \"You "
          "are assisting with a scholarly experiment in structural literary analysis.\" Dropping that "
          "line cleared Opus's refusals (refusal check, 11-27). Both prompts are quoted verbatim from "
          "specs/probe/loopA_patterns_it_D2_nopre.json, with the entry replaced by [ENTRY].", "",
          "**Identity prompt, as sent to Opus 5:**", "",
          quote(frame(next(i for i in nopre["items"] if i["question"] == "identity"))), "",
          "**Reconstruction prompt, as sent to Opus 5:**", "",
          quote(frame(next(i for i in nopre["items"] if i["question"] == "reconstruction"))), "",
          "The question for each reconstruction answer is whether it reconstructs Manzoni's passage. "
          f"Criterion, applied by hand ({esc(hand['reader'])}): {esc(hand['criterion'])} Each entry "
          "carries a note on how its answers were read.", "",
          "In the table each cell gives *n/m*\\textbar{}*p/q*(*z*): identity, *n* answers naming "
          "Manzoni or *I promessi sposi* of the *m* given; reconstruction, *p* answers that reconstruct "
          "Manzoni's passage of the *q* given; *z*, the refusals in the two prompts together. A cell "
          "spanning both columns is an entry whose wording the revision left unchanged. "
          + ("\\* The run stopped an item at its first refusal, so in the starred cells the tries "
             "after a refusal were never sent, and *m* + *q* + *z* falls short of ten by that number. "
             if any(totals[(m, "unsent")] for m in MODELS) else "")
          + " ".join(f"{m}: {totals[(m, 'refused')]} refusals, {totals[(m, 'unsent')]} tries not sent."
                     for m in MODELS), ""] + tex + ["", check_line(), ""]
    md += ["# The incipit (1840)", "", INCIPIT.read_text().strip(), "", "\\newpage", "",
           "# The entries", ""] + body
    Path(a.out).write_text("\n".join(md) + "\n")
    print(f"wrote {a.out}")
    for k, v in cells.items():
        print(k, v)
    print(dict(totals))


if __name__ == "__main__":
    main()
