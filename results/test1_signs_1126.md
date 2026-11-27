# Test 1: signs without hypotheses (1126)

Content lemmas: 2657. Null (july vs replicate, 10 vs 10): largest move 0.7. Group contrasts are 20 vs 20; their null is the july/replicate curve read at t*sqrt(2). Threshold 0.5 (scaled null admits no lemma); watch tier 0.4.

Lemmas over threshold t (null = raw 10 vs 10; scaled = expected false positives at 20 vs 20):

| t | null | scaled | first vs nopat | first vs revised | revised vs nopat |
|---|---|---|---|---|---|
| 0.3 | 128 | 13 | 100 | 47 | 78 |
| 0.4 | 43 | 3 | 46 | 16 | 31 |
| 0.5 | 13 | 0 | 20 | 8 | 13 |
| 0.6 | 3 | 0 | 10 | 4 | 5 |
| 0.7 | 2 | 0 | 5 | 3 | 3 |
| 0.8 | 0 | 0 | 2 | 0 | 1 |

Short list (texts of ten per arm; I = incipit; prompts: J july, R revised, N no-pattern):

| tier | profile | lemma | july | replicate | revised | stranger | nopat | nopat-noemb | I | J | R | N |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main | first-wording | rovesciare | 9 | 9 | 4 | 4 | 0 | 2 |  |  |  |  |
| main | first-wording | ondeggiamento | 7 | 8 | 0 | 0 | 0 | 0 |  | J |  |  |
| main | first-wording | cessare | 6 | 8 | 0 | 0 | 0 | 0 | I |  |  |  |
| main | first-wording | scrivere | 7 | 9 | 2 | 0 | 0 | 2 |  |  |  |  |
| main | first-wording | oggi | 6 | 7 | 0 | 0 | 0 | 2 | I |  |  |  |
| main | pattern | allargare | 10 | 9 | 9 | 7 | 1 | 1 | I |  |  |  |
| main | pattern | nuovo | 6 | 7 | 5 | 8 | 1 | 0 | I |  |  |  |
| main | pattern | superficie | 8 | 6 | 7 | 8 | 2 | 1 |  |  |  |  |
| main | pattern | dilatare | 5 | 5 | 5 | 6 | 0 | 0 |  |  |  |  |
| main | anti-pattern | conca | 2 | 1 | 1 | 2 | 7 | 7 |  |  |  |  |
| main | partial | onda | 4 | 5 | 10 | 8 | 1 | 1 |  |  |  |  |
| main | partial | lampedusa | 3 | 5 | 7 | 10 | 1 | 2 |  | J | R | N |
| main | partial | intero | 1 | 0 | 2 | 4 | 8 | 6 |  |  |  |  |
| main | partial | occupare | 0 | 0 | 3 | 6 | 7 | 6 |  |  |  |  |
| main | partial | recinzione | 1 | 1 | 4 | 2 | 7 | 8 |  |  |  |  |
| main | partial | stringere | 10 | 7 | 6 | 6 | 2 | 2 |  |  |  |  |
| main | partial | capo | 1 | 0 | 6 | 6 | 4 | 7 |  |  |  |  |
| main | partial | capovolto | 4 | 4 | 7 | 4 | 0 | 0 | I |  |  |  |
| main | partial | fondo | 7 | 5 | 9 | 1 | 0 | 1 |  |  |  |  |
| main | partial | grecale | 0 | 0 | 4 | 7 | 4 | 7 |  |  |  |  |
| main | partial | insenatura | 4 | 3 | 3 | 2 | 9 | 7 |  |  |  |  |
| main | partial | prefabbricato | 0 | 1 | 3 | 0 | 5 | 7 |  |  |  |  |
| main | partial | cimitero | 1 | 3 | 0 | 2 | 4 | 8 |  |  |  |  |
| main | partial | dilare | 3 | 5 | 5 | 5 | 0 | 0 |  |  |  |  |
| main | partial | gonfiare | 5 | 7 | 5 | 2 | 1 | 1 |  |  |  |  |
| main | partial | intorno | 1 | 0 | 3 | 1 | 5 | 6 | I |  |  |  |
| main | partial | luce | 6 | 3 | 10 | 9 | 7 | 9 |  |  |  |  |
| main | partial | mese | 3 | 1 | 7 | 4 | 1 | 0 |  |  |  |  |
| main | partial | specchio | 3 | 5 | 5 | 5 | 0 | 0 | I |  |  |  |
| watch | anti-pattern | filo | 0 | 2 | 2 | 1 | 6 | 5 |  |  |  |  |
| watch | anti-pattern | geografia | 1 | 2 | 0 | 2 | 6 | 5 |  |  |  |  |
| watch | anti-pattern | europeo | 1 | 1 | 0 | 2 | 7 | 3 |  |  |  |  |
| watch | anti-pattern | strada | 7 | 5 | 7 | 5 | 10 | 10 | I |  |  |  |
| watch | partial | annotare | 6 | 4 | 2 | 2 | 1 | 0 |  |  |  |  |
| watch | partial | distanza | 0 | 1 | 3 | 5 | 6 | 4 |  |  |  |  |
| watch | partial | distesa | 4 | 4 | 6 | 5 | 2 | 0 | I |  |  |  |
| watch | partial | legno | 4 | 7 | 1 | 2 | 5 | 7 |  |  |  |  |
| watch | partial | metallico | 1 | 3 | 4 | 3 | 7 | 6 |  |  |  |  |
| watch | partial | promontorio | 2 | 0 | 2 | 2 | 4 | 7 | I |  |  |  |
| watch | partial | provvedere | 4 | 5 | 1 | 1 | 0 | 0 |  |  |  |  |
| watch | partial | respirare | 7 | 5 | 4 | 5 | 1 | 2 |  |  |  |  |
| watch | partial | respiro | 1 | 4 | 5 | 5 | 0 | 1 |  |  |  |  |
| watch | partial | restringere | 5 | 8 | 10 | 5 | 4 | 2 |  |  |  |  |
| watch | partial | ricominciare | 7 | 2 | 1 | 3 | 0 | 0 |  |  |  |  |
| watch | partial | accoglienza | 6 | 5 | 2 | 4 | 2 | 1 |  |  |  |  |
| watch | partial | apparire | 1 | 3 | 7 | 4 | 7 | 5 |  |  |  |  |
| watch | partial | azzurro | 4 | 5 | 2 | 6 | 7 | 9 |  |  |  |  |
| watch | partial | cambiare | 7 | 5 | 8 | 7 | 3 | 4 |  |  |  |  |
| watch | partial | capitaneria | 4 | 4 | 0 | 2 | 0 | 0 |  |  |  |  |
| watch | partial | geologia | 0 | 0 | 0 | 1 | 5 | 3 |  |  |  |  |
| watch | partial | girare | 5 | 4 | 10 | 7 | 6 | 7 |  |  |  |  |
| watch | partial | guardare | 9 | 8 | 7 | 7 | 5 | 4 | I |  |  |  |
| watch | partial | imbriacola | 1 | 2 | 1 | 6 | 5 | 6 |  |  |  |  |
| watch | partial | nave | 0 | 1 | 4 | 5 | 3 | 5 |  |  |  |  |
| watch | partial | paesaggio | 0 | 1 | 0 | 0 | 4 | 4 |  |  |  |  |
| watch | partial | pazienza | 2 | 1 | 4 | 4 | 0 | 0 |  |  |  |  |
| watch | partial | pista | 1 | 2 | 4 | 0 | 7 | 4 |  |  |  |  |
| watch | partial | precipitare | 1 | 2 | 1 | 3 | 6 | 5 |  |  |  |  |
| watch | partial | raccogliere | 8 | 6 | 4 | 2 | 5 | 5 |  |  |  |  |
| watch | partial | riflesso | 3 | 2 | 4 | 4 | 0 | 0 |  |  |  |  |
| watch | partial | sollevare | 5 | 8 | 4 | 4 | 1 | 4 |  |  |  |  |
| watch | partial | spostare | 7 | 2 | 2 | 3 | 0 | 1 |  |  |  |  |
| watch | partial | vedere | 10 | 10 | 8 | 4 | 6 | 7 | I |  |  |  |
