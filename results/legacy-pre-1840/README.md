# Runs made before 2026-07-30 — superseded twice, kept for identification

Everything in this directory was generated and scored **against the wrong source text**, and under
the vocabulary the D2/D3 relabel retired. Neither its numbers nor its filenames should be used.

**Wrong source.** Until 2026-07-30 the Italian pipeline parsed a string literal in
`scripts/run_production.py`, not the 1840 quarantana this project declares. It read *"verso
l'estremità da un'ampia montagna … in un grande specchio rovesciati coll'immagine ripetuti"* and
*"sul declivio"* where the 1840 text reads *"o piuttosto smarrito in un gruppo, in un andirivieni di
montagne … e che l'acqua riflette capovolti, co' paesetti posti sulle rive"* and *"sulla costa"*.
Periods 1–7 parse identically; **period 8** had 11 clauses at depth 6 against the correct 14 at depth
5, and **period 9** depth 5 against 4. The archived text is `texts/incipit_it_as_parsed.txt`
(sha1 `4a32836dc1c1`); the live source is `texts/incipit_it.txt` (`0d3a92484994`).

**Retired vocabulary.** `d2a` and `d2c` in these filenames, and the `grain` field inside them, are
the old two-grain reading of a single D2 rung: **grain a is now D3 (clause inventory), grain c is now
D2 (clause nesting)**, and what used to be D3 is now D4. The scorers still read the old keys.

Live runs are named `production_<degree>_<lang>_<tag>_<model>.json` and record `source_ref` +
`source_sha1` instead of the passage itself.
