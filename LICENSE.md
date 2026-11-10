# Licenses

This repository is **dual-licensed**: the **content** (the rubric data, specs, generated
texts, and prose documentation) and the **software** (the pipeline scripts) are released
under different terms appropriate to each.

Copyright (c) 2026 Manuela Marchesini and Stefano Franchi.

---

## Content — `spec/`, `results/`, `texts/`, `docs/`, `queneau/Experiments/`

**License**: Creative Commons Attribution-ShareAlike 4.0 International
**SPDX identifier**: `CC-BY-SA-4.0`
**Canonical text**: see [`LICENSE-docs`](LICENSE-docs) in this repository
**Online reference**: https://creativecommons.org/licenses/by-sa/4.0/legalcode

Covers the rubric (`spec/patterns.json`, `constraints*.json`), the transposition brief
(`spec/brief_*.json`), the built form-targets (`spec/specs_it_*.json`), the generated texts and scored outputs in
`results/`, and the prose documentation in `docs/`. You are free to share and adapt this
material — including commercially — under the following terms:

- **Attribution** — give appropriate credit, link to the license, and indicate if changes
  were made. Suggested citation in `README.md`.
- **ShareAlike** — if you remix, transform, or build upon the material, you must distribute
  your contributions under the same license as the original.

The Italian source text in `texts/` is from Manzoni, *I promessi sposi* (1840), and is in the
**public domain**. The English incipit (revised Colquhoun translation) and the Queneau passage
with its English and Italian translations, in `queneau/Experiments/Queneau/queneau_prototype/`,
are in copyright and are **not** covered by this license: they are short excerpts quoted for
scholarly analysis (see the README).

## Software — `scripts/*.py`, `scripts/*.sbatch`, `queneau/scripts/`

**License**: GNU General Public License version 3
**SPDX identifier**: `GPL-3.0-only`
**Canonical text**: see [`LICENSE-code`](LICENSE-code) in this repository
**Online reference**: https://www.gnu.org/licenses/gpl-3.0.txt

Covers all the pipeline software: `build_rubric.py`, `build_specs.py`, `run_homeo.py`,
`run_production.py`, `score_production.py`, `tree_metric.py`, `sem_metrics.py`,
`overlap_metrics.py`, `monitor.py`, and the ACES SLURM scripts. You may redistribute and
modify the software under the terms of the GPL-3.0; modified versions must remain under the
same license, and redistribution must include the source.

## Documentation — `README.md`, `MANIFEST.md`, `LICENSE.md`, `.gitignore`

CC-BY-SA-4.0, same terms as the content.

---

## Per-file SPDX headers

For tooling that scans license metadata via the [REUSE specification](https://reuse.software/),
the defaults are:

- `scripts/*.py`, `scripts/*.sbatch`, `queneau/scripts/**`: `SPDX-License-Identifier: GPL-3.0-only`
- `spec/**`, `results/**`, `docs/**`, `queneau/Experiments/**`, `*.md`: `SPDX-License-Identifier: CC-BY-SA-4.0`

Per-file headers are not yet inserted; the declarations in this `LICENSE.md` are authoritative
for the v1 deposit.

---

## Choice of licenses — rationale

- **CC-BY-SA-4.0 for content**: standard for open data and scholarly archives. Permits
  commercial use and modification under attribution; the ShareAlike clause keeps derivative
  works in the open commons. Compatible with academic citation norms.
- **GPL-3.0 for code**: ensures the pipeline remains open source — modifications must be
  published under the same license. Matches the licensing of the sibling repositories, giving
  the whole research programme a consistent legal footing.
