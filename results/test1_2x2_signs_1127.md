# Test 1 on the 2x2 dataset (1127)

Arms (ten texts each): **pat+emb** revised patterns with the embargo; **pat** revised patterns, no embargo (the reading arm); **emb** no patterns, embargo; **none** no patterns, no embargo. Cells: texts (of ten) containing the lemma.

Content lemmas: 2268. Null: 2000 stratified shuffles (seed 20261127). Tiers fixed before reading: firm = smallest t with family-wise rate <= 0.05; candidate = smallest t with fewer than one chance pass expected.

Flags: I incipit, E embargo block, P pattern block, B prompt shared by all four (P and B are lemma matches: the prompts are in English, so they catch only Italian words in them); by hand, T renders an English word of the pattern block, U one of the shared prompt (the English is in the last column).

## patterns: pat+emb + pat against emb + none

| t | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|
| expected by chance | 5.53 | 0.69 | 0.07 | 0.00 | 0.00 | 0.00 | 0.00 |
| family-wise rate | 0.99 | 0.49 | 0.07 | 0.00 | 0.00 | 0.00 | 0.00 |
| observed | 31 | 16 | 8 | 2 | 1 | 0 | 0 |

Firm at t >= 0.6, candidate at t >= 0.4.

| tier | lemma | dir | score | pat+emb | pat | emb | none | flags | renders |
|---|---|---|---|---|---|---|---|---|---|
| firm | onda | up | 0.7 | 10 | 8 | 1 | 1 |  |  |
| firm | allargare | up | 0.6 | 9 | 7 | 1 | 1 | IT | P5-01 expansion, re-expansion |
| candidate | conca | down | 0.5 | 1 | 2 | 7 | 7 |  |  |
| candidate | dilare | up | 0.5 | 5 | 5 | 0 | 0 | T | P5-01 dilation (stanza's lemma for some forms of dilatare) |
| candidate | dilatare | up | 0.5 | 5 | 6 | 0 | 0 | T | P5-01 dilation |
| candidate | lampedusa | up | 0.5 | 7 | 10 | 1 | 2 | B |  |
| candidate | specchio | up | 0.5 | 5 | 5 | 0 | 0 | IT | P6-01 specular, mirroring medium |
| candidate | superficie | up | 0.5 | 7 | 8 | 2 | 1 |  |  |
| candidate | capovolto | up | 0.4 | 7 | 4 | 0 | 0 | IT | P6-01 inversion marker |
| candidate | insenatura | down | 0.4 | 3 | 2 | 9 | 7 |  |  |
| candidate | nuovo | up | 0.4 | 5 | 8 | 1 | 0 | IT | P5-01 re-expansion (di nuovo); also 'NEW material' |
| candidate | paesaggio | down | 0.4 | 0 | 0 | 4 | 4 | U | domain brief: migration landscape |
| candidate | pazienza | up | 0.4 | 4 | 4 | 0 | 0 |  |  |
| candidate | respiro | up | 0.4 | 5 | 5 | 0 | 1 |  |  |
| candidate | riflesso | up | 0.4 | 4 | 4 | 0 | 0 | T | P6-01 reflection predicate |
| candidate | stringere | up | 0.4 | 6 | 6 | 2 | 2 | T | P5-01 constriction |

## embargo: pat+emb + emb against pat + none

| t | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|
| expected by chance | 4.96 | 0.62 | 0.07 | 0.00 | 0.00 | 0.00 | 0.00 |
| family-wise rate | 0.99 | 0.46 | 0.07 | 0.00 | 0.00 | 0.00 | 0.00 |
| observed | 8 | 1 | 0 | 0 | 0 | 0 | 0 |

Firm at t >= 0.6, candidate at t >= 0.4.

| tier | lemma | dir | score | pat+emb | pat | emb | none | flags | renders |
|---|---|---|---|---|---|---|---|---|---|
| candidate | chiaro | up | 0.4 | 7 | 2 | 7 | 3 |  |  |

## patterns|no emb: pat against none

| t | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|
| expected by chance | 125.77 | 40.29 | 11.34 | 2.62 | 0.49 | 0.07 | 0.01 |
| family-wise rate | 1.00 | 1.00 | 1.00 | 0.93 | 0.38 | 0.06 | 0.01 |
| observed | 190 | 72 | 33 | 13 | 6 | 2 | 0 |

Firm at t >= 0.9, candidate at t >= 0.7.

| tier | lemma | dir | score | pat+emb | pat | emb | none | flags | renders |
|---|---|---|---|---|---|---|---|---|---|
| candidate | lampedusa | up | 0.8 | 7 | 10 | 1 | 2 | B |  |
| candidate | nuovo | up | 0.8 | 5 | 8 | 1 | 0 | IT | P5-01 re-expansion (di nuovo); also 'NEW material' |
| candidate | correre | down | 0.7 | 3 | 0 | 3 | 7 | I |  |
| candidate | onda | up | 0.7 | 10 | 8 | 1 | 1 |  |  |
| candidate | prefabbricato | down | 0.7 | 3 | 0 | 5 | 7 |  |  |
| candidate | superficie | up | 0.7 | 7 | 8 | 2 | 1 |  |  |

## patterns|emb: pat+emb against emb

| t | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|
| expected by chance | 123.56 | 42.14 | 10.78 | 2.80 | 0.45 | 0.07 | 0.01 |
| family-wise rate | 1.00 | 1.00 | 1.00 | 0.94 | 0.35 | 0.06 | 0.01 |
| observed | 181 | 76 | 28 | 13 | 5 | 3 | 2 |

Firm at t >= 0.9, candidate at t >= 0.7.

| tier | lemma | dir | score | pat+emb | pat | emb | none | flags | renders |
|---|---|---|---|---|---|---|---|---|---|
| firm | fondo | up | 0.9 | 9 | 1 | 0 | 1 |  |  |
| firm | onda | up | 0.9 | 10 | 8 | 1 | 1 |  |  |
| candidate | allargare | up | 0.8 | 9 | 7 | 1 | 1 | IT | P5-01 expansion, re-expansion |
| candidate | capovolto | up | 0.7 | 7 | 4 | 0 | 0 | IT | P6-01 inversion marker |
| candidate | europeo | down | 0.7 | 0 | 2 | 7 | 3 | U | domain brief: Italian and European policing apparatus |

## embargo|patterns: pat+emb against pat

| t | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|
| expected by chance | 121.31 | 37.74 | 10.81 | 2.53 | 0.47 | 0.06 | 0.01 |
| family-wise rate | 1.00 | 1.00 | 1.00 | 0.91 | 0.37 | 0.06 | 0.01 |
| observed | 169 | 58 | 23 | 4 | 4 | 1 | 0 |

Firm at t >= 0.9, candidate at t >= 0.7.

| tier | lemma | dir | score | pat+emb | pat | emb | none | flags | renders |
|---|---|---|---|---|---|---|---|---|---|
| candidate | fondo | up | 0.8 | 9 | 1 | 0 | 1 |  |  |
| candidate | cale | up | 0.7 | 9 | 2 | 5 | 5 |  |  |
| candidate | esistere | down | 0.7 | 2 | 9 | 4 | 5 |  |  |
| candidate | recinto | up | 0.7 | 7 | 0 | 6 | 6 |  |  |

## embargo|no pat: emb against none

| t | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|
| expected by chance | 126.46 | 41.10 | 11.59 | 2.64 | 0.51 | 0.06 | 0.01 |
| family-wise rate | 1.00 | 1.00 | 1.00 | 0.93 | 0.40 | 0.06 | 0.01 |
| observed | 139 | 38 | 8 | 1 | 0 | 0 | 0 |

Firm at t >= 0.9, candidate at t >= 0.7.

No lemma reaches a tier.

Text lemmas identical to the 11-26 parse: True.
