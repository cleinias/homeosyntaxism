# The target-domain brief — `spec/brief_<lang>.json`

The **brief** is the only place the *transposition domain* is decided: the source-blind
scene-world the generator must build while holding the source's syntactic form. Everything
else in the pipeline (the degree target, the patterns, the embargo) is about *form* or
*fidelity*; the brief is the *content* you are transposing **into**.

It is read by `build_specs.py`, compiled into each `spec/specs_<lang>_<degree>.json`, and
shown to the model under the prompt's `TARGET DOMAIN` heading. The generator is told to
deploy these elements **in its own words** — the brief is a functional description of a
world, **not** a word-for-word substitution table.

## File format

One JSON object per language, e.g. `spec/brief_it.json`:

```json
{
  "lang": "it",
  "domain": "the border/migration landscape of the central Mediterranean",
  "elements": [
    "Quel braccio di mare tra la costa libica e Lampedusa (il Canale di Sicilia)",
    "il Faro di Capo Grecale",
    "a Lampedusan fisherman at his nets, at dawn, on the eve of a sighting"
  ]
}
```

| field | what it is |
|-------|------------|
| `lang` | `"it"` or `"en"`. **Must match the `--lang` you build with** — `build_specs.py` exits with an error on mismatch (the file is language-specific). |
| `domain` | a short noun phrase naming the world; goes verbatim into the prompt's `TARGET DOMAIN — …` line. |
| `elements` | the scene-world items, one per list entry; rendered as a bulleted brief in the prompt. |

## What goes into an `element`

Each element is one **functional building block** of the target world. Good elements are:

- **Concrete and deployable** — a place, an object, an actor, an institution, a juridical
  zone, a recurring image (`il Faro di Capo Grecale`; `the charted SAR boundary`; `a
  fisherman at his nets`). The model should be able to *put it in the scene*.
- **Source-blind** — describe the *target* world on its own terms. Do **not** phrase an
  element as "the equivalent of Manzoni's X"; the generator never sees the source.
- **Free to carry guidance, in parentheses** — register, what to avoid, or a tie to a
  formal pattern, e.g. `"an equivalent now-deixis (oggigiorno / ai giorni nostri — NOT 'al
  giorno d'oggi', which is embargoed)"`. This is how you steer a pattern onto new material.
- **Mixed-language is fine** — proper names and set phrases can stay in the target language
  even in an otherwise-English gloss; the model reads both.

You can have as many or as few elements as the passage needs; order is preserved but not
binding. To **retarget** to a different domain, edit `domain` + `elements` (or write a new
`brief_<lang>.json`), then re-run `build_specs.py` for each degree you use.

## How it relates to the research provenance

In the research project the Italian elements were *derived* from a four-column
`source → target` correspondence table (`source_element, source_function, target_element,
justification`) that records, for the paper's audit trail, **why** each target was chosen
to bear a given critical claim. Only the `target_element` column ever reached the model.
The public tool keeps just that surviving column, as this brief — the rich correspondence
table stays in the private research repo.

## Future: author as a spreadsheet

A planned convenience: let authors edit the brief as a **CSV** (one element per row, plus
optional columns for the pattern it supports and authoring notes) and convert it to
`brief_<lang>.json` with a small `brief_from_csv.py` helper. Not yet implemented; the JSON
above is the current source of truth.
