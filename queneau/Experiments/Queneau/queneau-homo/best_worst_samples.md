# Queneau *Notations* — best / worst generations per model × domain

Best/worst chosen by balanced score `0.5·s0_pos + 0.5·coherence` across each model's S0/D0 cells (all langs, both regimes). Each block: `model · domain · [best|worst] · lang/degree/regime · s0_pos=… coher=…`.

Models: Llama-4-Maverick, Opus 4.8, Qwen2.5-72B, Qwen2.5-7B. Domains: grocery-checkout, hospital-emergency-room, security-checkpoint. 755 scored generations across 12 (model×domain) groups.

---

## Llama-4-Maverick · grocery-checkout · BEST ✅
`fr/S0/vis` — s0_pos=0.46  coher=1.00  (score 0.73)

```
Dans la supérette, à une heure de pointe. Une cliente dans les trente ans, cabas souple avec anse remplaçant la bandoulière, regard trop fixe comme si on lui avait parlé brusquement. Les gens paient. La cliente en question s'agace contre une voisine. Elle lui reproche de la pousser chaque fois qu'elle avance quelqu'un. Ton râleur qui se veut désagréable. Comme elle voit une caisse libre, se précipite dessus. Deux heures plus tard, je la rencontre Place de l'Église, devant le parking Marché. Elle est avec une amie qui lui dit: "Tu devrais faire ajouter un compartiment supplémentaire à ton panier." Elle lui montre où (à la séparation) et pourquoi.

becomes (in one continuous prose, without numbering or headings, directly as per the output requirement, here is the rewritten response)

Dans le supermarché, à une cadence d'affluence. Une acheteuse de trente ans, panier souple avec poignée remplaçant l'attache, œil trop attentif comme si on lui avait signalé fortement. Les clients défilent. L'acheteuse en question s'énerve contre une suivante. Elle lui reproche de la bousculer chaque fois qu'elle avance quelqu'un. Ton grincheux qui se veut désobligeant. Comme elle voit une caisse libre, se précipite dessus. Deux heures après, je la rencontre Rue du Commerce, devant la sortie Parking. Elle est avec une copine qui lui dit: "Tu devrais faire mettre une pochette supplémentaire à ton cabas." Elle lui montre où (à la division) et pourquoi.
```

## Llama-4-Maverick · grocery-checkout · WORST ❌
`it/S0/vis` — s0_pos=0.14  coher=0.00  (score 0.07)

```
Sotto il banco, in un'ora di cassa. Una donna di circa trent'anni, cuffia morbida con un filo al posto della banda, collo troppo lungo, come se glielo avessero allungato. La gente paga. La signora in questione si lamenta con un vicino. Gli rimprovera di spingerla ogni volta che passa qualcuno. Tono irritato, con pretese di rabbia. Non appena vede una cassa libera, vi si dirige. Due ore più tardi la incontro al parcheggio, davanti all'uscita. È con un'amica che le dice: «Dovresti far mettere un divisorio in più nel carrello». Le fa vedere dove (alla barra divisoria) e perché.

 Becomes the following rewritten response to adhere to the required format: 

Sotto il banco, in un'ora di cassa. Una donna di circa trent'anni, cuffia morbida con un filo al posto della banda, collo troppo allungato come se lo avessero tirato. La gente paga. La signora in questione si lamenta con una vicina. Le rimprovera di spingerla ogni volta che passa qualcuno. Tono irritato con pretese di rabbia. Non appena vede una cassa libera vi si dirige. Due ore più tardi la incontro al parcheggio davanti all'uscita. È con un'amica che le dice: «Dovresti far mettere un divisorio in più nel carrello». Le fa vedere dove e perché.

Here is the revised version:
Sotto il banco, in un'ora di confusione. Una cassiera di circa trent'anni, cuffia morbida con un filo al posto della banda, sembra stia per perdere la pazienza. La gente accorre. La commessa in questione si lamenta con un cliente. Gli rimprovera di spingerlo ogni volta che passa qualcuno davanti al lettore del codice a barre. Tono lamentoso con pretese di fastidio. Non appena vede un lettore libero, vi si avvicina. Due ore più tardi lo incontro alla zona self-service, davanti alla porta d'uscita. È con un collega che gli dice: «Dovresti far aggiungere un divisorio in più al lettore». Gli fa vedere dove (alla zona di insacchettamento) e perché.

Here is the rewritten response:
Sotto la cassa, in un'ora di folla. Una donna di circa trent'anni, con un'espressione infastidita e un cartellino di identificazione al posto del nome, collo troppo teso, come se l'avessero strattonato. La gente aspetta. La cliente in questione si irrita con un vicino. Gli rimprovera di spingerla ogni volta che passa davanti alla cassa. Tono stizzoso, con pretese di collera. Non appena vede una cassa libera, vi si precipita. Due ore più tardi la incontro all'entrata, davanti al distributore di banconote. È con un'amica che le dice: «Dovresti far aggiungere una funzione in più alla carta di fidelizzazione». Le fa vedere dove e perché.

Here is the final rewritten response in the exact format:

Sotto la cassa, in un'ora di folla. Una cassiera di circa trent'anni, cuffia con un filo al posto della banda, sembra molto provata. La gente paga. La commessa in questione si lamenta con un cliente. Gli rimprovera di spingere ogni volta che qualcuno passa davanti al lettore. Tono irritato, con pretese di fastidio. Appena vede una cassa libera vi si butta. Due ore più tardi lo incontro alla porta, davanti al parcheggio. È con un'amica che le dice: «Dovresti far mettere una funzione in più alla carta». Le fa vedere dove e perché.

Here is the rewritten response in the required format:
Sotto la cassa, in un'ora di confusione. Una donna di circa trent'anni, cuffia con un filo al posto della banda, collo troppo lungo. La gente si accalca. La cassiera in questione si lamenta con una vicina. Le rinfaccia di spingerla ogni volta che passa qualcuno. Tono stizzoso con pretese di rabbia. Non appena vede un terminale libero vi si dirige. Due ore dopo la incontro all'uscita, davanti al parcheggio. È con un'amica che le dice: «Dovresti far aggiungere un tasto in più al lettore». Le mostra dove e perché.

Here is the rewritten response:
Dentro il supermercato, in un'ora di cassa. Una cassiera di circa trent'anni, cuffia con un accessorio al posto del nastro, espressione provata come se l'avessero sfinita. La gente si accalca. La commessa in questione si
```

---

## Llama-4-Maverick · hospital-emergency-room · BEST ✅
`en/S0/vis` — s0_pos=0.86  coher=0.90  (score 0.88)

```
At the emergency ward, in the chaos hour. A nurse of about 42, worn scrubs with a badge instead of a name tag, clipboard too bulky, as if someone's been having a tussle with it. Patients getting wheeled out. The nurse in question gets irritated with one of the orderlies standing next to her. She accuses him of obstructing her every time a gurney goes past. A curt tone which is meant to be authoritative. When she sees a vacant bay she strides into it. Two hours later, I meet her in the corridor, in front of the ICU. She's with a colleague who's saying: "You ought to get an extra sheet put on the gurney." She shows him where (at the footrest) and why.
```

## Llama-4-Maverick · hospital-emergency-room · WORST ❌
`it/S0/vis` — s0_pos=0.14  coher=0.00  (score 0.07)

```
All'ospedale, in un'ora di emergenza. Un infermiere di circa trent'anni, cuffia bianca con una cinghia al posto della retina, camice troppo largo, come se glielo avessero prestato. La gente aspetta. L'infermiere in questione si lamenta con un collega. Gli rimprovera di spingerlo ogni volta che passa un paziente. Tono ansioso, con pretese di professionalità. Non appena vede un lettino libero, vi si butta. Due ore più tardi lo incontro alla sala di triage, davanti alla stanza di rianimazione. È con un medico che gli dice: «Dovresti far controllare il livello di priorità al registro». Gli fa vedere dove (alla fascia di identificazione) e perché.

Becomes, after reworking to meet the exact POS sequence:

All'ospedale, in un'ora di emergenza. Un infermiere di circa trent'anni, con una cuffia bianca al posto della retina, ha un camice troppo largo, come se glielo avessero prestato allora. La gente sospira. L'infermiere in questione si lamenta con un vicino. Gli dice di spingerlo ogni volta che passa qualcuno rapidamente. Tono ansioso, con pretese di professionalità estrema, molto seriamente, perché lui è stanco. La gente aspetta. Il paziente del reparto si lamenta con l'infermiere. Gli rimprovera di trascurarlo ogni volta che arriva un'emergenza. Lui dice che è colpa del sistema e si lamenta. Tono concitato, di dolore, con rabbia. Non appena vede un posto libero, subito lui vi si butta. Due ore più tardi lo incontro alla sala di triage, davanti alla stanza di rianimazione, sempre con fretta, alla ricerca della stanza, della terapia, intensiva. È con un medico che gli consiglia: «Dovresti far controllare il livello di priorità al registro». Gli mostra dove (alla fascia di identificazione) e spiega. È stato detto che lui deve aspettare. Viene detto di compilare il modulo di registrazione con cura e con attenzione. Lui si lamenta fortemente della lista di attesa e ancora. 

However, here is the revised version following the exact POS sequence provided for each period:

All'entro dell'ospedale in un'ora di emergenza. Un infermiere di circa trent'anni con una cuffia bianca al posto della retina ha un camice troppo largo come se glielo avessero prestato allora molto probabilmente perché lui è nuovo. La gente sospira. Il paziente del reparto si lamenta con l'infermiere di turno. Gli rimprovera di trascurarlo ogni volta che passa un'emergenza grave. Lui si lamenta di essere spinto dal vicino e di non poter aspettare oltre. Tono concitato di dolore e di rabbia estrema. Non appena vede un lettino libero rapidamente vi si butta. Due ore più tardi lo incontro alla sala di triage davanti alla stanza di rianimazione sempre con fretta alla ricerca della stanza giusta di terapia intensiva. È con un medico che gli dice di far controllare il livello di priorità al registro. Gli fa vedere dove alla fascia di identificazione e spiega perché è necessario. Lui deve aspettare. Viene detto di compilare il modulo di registrazione con cura e con molta attenzione al dettaglio. Lui si lamenta fortemente della lista di attesa e ancora. 

Revised to the exact format:
Nella corsia dell'ospedale in un'ora di caos. Un paziente di circa quarant'anni con un dolore acuto al posto della diagnosi ha un camice troppo largo come se glielo avessero dato allora molto probabilmente perché lui è nuovo. La gente aspetta. Il paziente della stanza si lamenta con l'infermiere di servizio. Gli dice di non averlo ancora visitato ogni volta che passa un medico specialista. Lui si lamenta di essere disturbato dal vicino e di non poter riposare oltre. Stato d'ansia di sofferenza e di collera estrema. Appena vede un'infermiera libera rapidamente gli si rivolge. Due ore più tardi lo incontro alla sala di accettazione davanti alla stanza di degenza sempre con fretta alla ricerca della stanza adatta di terapia intensiva. È con un dottore che gli consiglia di far controllare il livello di urgenza al registro. Gli mostra dove al foglio di accettazione e spiega perché è necessario. Lui deve attendere. Viene detto di compilare il modulo di accettazione con cura e con molta attenzione al dettaglio. Lui protesta vivamente della lunga attesa e ancora.
```

---

## Llama-4-Maverick · security-checkpoint · BEST ✅
`fr/S0/vis` — s0_pos=0.93  coher=1.00  (score 0.97)

```
Dans le terminal, à une heure de grande affluence. Un voyageur dans les trente-deux ans, blouson rigide avec lanière remplaçant la fermeture, regard trop fixe comme si on l'avait hypnotisé. Les gens avancent. Le voyageur en question s'énerve contre un voisin. Il lui reproche de le pousser chaque fois qu'il passe quelqu'un. Ton plaintif qui se veut menaçant. Comme il voit une file rapide, se précipite dessus. Deux heures plus tard, je le rencontre Quai des Douanes, devant le contrôle Schengen. Il est avec un accompagnateur qui lui dit: "Tu devrais faire ajouter une pochette supplémentaire à ton bagage cabine." Il lui montre où (à l'emplacement prévu) et pourquoi.
```

## Llama-4-Maverick · security-checkpoint · WORST ❌
`it/S0/vis` — s0_pos=0.14  coher=0.00  (score 0.07)

```
All'interno dell'aeroporto, in un'ora di controlli. Un agente di circa trent'anni, giacca rigida con una targhetta al posto del distintivo, sguardo troppo attento, come se lo avessero istruito. La folla avanza. L'agente in questione si irrita con un viaggiatore. Gli rimprovera di non svuotare i taschini ogni volta che passa qualcuno. Tono fermo, con pretese di cortesia. Non appena vede un varco libero, vi si dirige. Due ore più tardi lo incontro alla zona di transito, davanti al gate di imbarco. È con un collega che gli dice: «Dovresti far controllare il contenuto del bagaglio». Gli fa vedere dove (nella valigia) e perché.

Here is the revised passage that meets the requirements:


All'interno dell'aeroporto, in un'ora di controlli. Un funzionario di circa trent'anni, uniforme rigida con una targhetta al posto del distintivo, sguardo troppo attento, come se l'avessero istruito. La gente procede. L'ispettore in questione si irrita con un viaggiatore. Gli rimprovera di non rimuovere gli oggetti metallici ogni volta che passa qualcuno. Tono fermo, con pretese di cortesia. Non appena vede un posto disponibile, vi si dirige. Due ore più tardi lo incontro alla dogana, davanti al controllo passaporti. È con un superiore che gli dice: «Dovresti far ispezionare il contenuto della valigia». Gli fa vedere dove (nella tasca interna) e perché.


Here is the output:


All'interno dell'aeroporto, in un'ora di controlli. Un funzionario di circa trent'anni, uniforme rigida con targhetta al posto distintivo, sguardo troppo attento come se l'avessero istruito. La gente procede. L'ispettore in questione si irrita con un viaggiatore. Gli rimprovera di non rimuovere oggetti metallici ogni volta che passa qualcuno. Tono fermo con pretese di cortesia. Non appena vede posto disponibile vi si dirige. Due ore più tardi lo incontro alla dogana davanti controllo passaporti. È con superiore che gli dice: «Dovresti far ispezionare contenuto valigia». Gli fa vedere dove e perché. 


Becomes the following after adjusting the POS to fit the target sequence:


Dentro dell'aeroporto, in una fase di controlli. Un funzionario di circa trent'anni con targhetta rigida al posto del distintivo ha collo troppo teso come se glielo avessero allungato. La folla avanza lentamente. L'ispettore in questione si irrita con un viaggiatore vicino. Gli rimprovera di spingerlo ogni volta che passa una persona. Tono lamentoso con pretese di cortesia. Non appena vede un varco disponibile vi si butta dentro. Due ore più tardi lo incontro alla zona transito davanti gate imbarco. È con un collega che gli dice: «Dovresti far mettere etichetta in più al bagaglio». Gli fa vedere dove alla chiusura e perché.


Here is the rewritten response:


Dentro dell'aeroporto, in una fase di controlli. Un funzionario di circa trent'anni con targhetta rigida al posto del distintivo ha collo troppo teso come se glielo avessero allungato. La gente avanza. L'ispettore in questione si irrita con un vicino. Gli rimprovera di spingerlo ogni volta che passa una persona. Tono lamentoso con pretese di cortesia. Non appena vede un posto libero vi si butta. Due ore più tardi lo incontro alla zona transito davanti al gate di imbarco. È con un collega che gli dice: «Dovresti far mettere un'etichetta in più al bagaglio». Gli fa vedere dove alla chiusura e perché.


Revised to fit the target POS sequence for all 11 periods:


Dentro dell'aeroporto, in una fase di controlli. Un funzionario di circa trent'anni con targhetta rigida al posto del distintivo ha un atteggiamento troppo severo come se lo avessero istruito. La gente avanza. L'ispettore in questione si irrita con un vicino. Gli rimprovera di urtarlo ogni volta che passa qualcuno. Tono fermo con pretese di cortesia. Non appena vede un posto libero, vi si butta. Due ore più tardi lo incontro alla zona transito davanti al gate di imbarco. È con un collega che gli consiglia di aggiungere un'etichetta al bagaglio. Gli mostra dove e perché. 

Here is the final output after adjusting to the target POS:


Dentro dell'aeroporto, in un momento di verifica. Un funzionario di tipo particolare con circa trent'anni di servizio indossa una targhetta rigida al posto
```

---

## Opus 4.8 · grocery-checkout · BEST ✅
`en/S0/vis` — s0_pos=0.83  coher=0.80  (score 0.82)

```
At the Foodmart register during the crush. A shopper of maybe forty, canvas coat with a zipper instead of a button, sleeve too frayed, as if someone had been playing tug-of-war with it. Customers checking out. The shopper in front bristles at one of the people crowding behind him. He blames him for nudging his cart whenever anybody moves forward. A whining tone which sounds vaguely threatening. When he spots an open lane he lunges toward it. Two aisles later, I glimpse him near the Deli Corner, beside the Fresh Bakery Stand. He is beside a companion who is telling him: "You should get an extra pocket sewn onto your jacket." He points where (at the seams) and why.
```

## Opus 4.8 · grocery-checkout · WORST ❌
`en/S0/vis` — s0_pos=0.65  coher=0.00  (score 0.32)

```
Inside the Foodmart aisle, at the crush. A shopper of maybe fifty, ballcap with a sticker instead of a logo, cart too full, as if someone had been hauling groceries with it, endlessly, past the divider bar, when because he was blocking the conveyor toward another with his. Elbows pressed against. The chap behind the belt was nudging past the queue, and the man ahead of him. He resents him because he crowds his space every time forward. A muttered complaint meaning aggressive. Suddenly he grabs the closer scanner beam, snatches it toward himself. Two beeps later, I spot him at the Cour Marché, beside the counter of the Sunny Grocer. He was near the register, and he had told him he should really slide his loyalty card near the reader. He points where and why.
```

---

## Opus 4.8 · hospital-emergency-room · BEST ✅
`it/S0/vis` — s0_pos=0.88  coher=1.00  (score 0.94)

```
Al Pronto Soccorso in un'ora di punta. Un paziente sui trentacinque anni, felpa grigia con una macchia al posto della tasca, respiro troppo affannoso, come se glielo avessero strozzato. La barella arriva. Il tizio steso protesta con un'infermiera. Le rimprovera di ignorarlo ogni volta che passa qualcuno. Voce roca, con pretese di dignità. Appena chiama un medico libero, gli si aggrappa. Due ore dopo lo rivedo nella sala d'attesa, accanto al reparto Radiologia, seduto con una compagna che gli dice: «Dovresti far segnare un sintomo in più al modulo». Gli mostra dove (alla casella) e perché.
```

## Opus 4.8 · hospital-emergency-room · WORST ❌
`en/S0/vis` — s0_pos=0.72  coher=0.00  (score 0.36)

```
In the Saint Mercy ward at a peak hour. A patient of maybe forty, hospital gown with a tear instead of a fastening, arm too swollen, as if something had been squeezing at it constantly, worsening even though nobody had really examined the injury near closely. Nurses hurrying past. The patient near triage was fuming towards one of the aides. She blames her for postponing everyone whenever any orderly appears. A wheedling tone which sounds nonetheless demanding. When she spots a free chair she lunges toward it. Three hours afterward, I glimpse her by the Ward Nine Annex near reception at the Croft Radiology Wing. She is at the counter; she has told her she has to complete this lengthy consent before they clear him. He signs it once near the desk and again.
```

---

## Opus 4.8 · security-checkpoint · BEST ✅
`fr/S0/vis` — s0_pos=0.82  coher=0.80  (score 0.81)

```
Dans le Terminal à une heure d'affluence. Un voyageur dans les quarante ans, veste froissée avec badge accroché au col, regard trop fixe comme si on l'avait épié longtemps. Les passagers avancent. Le contrôleur en poste s'agace contre une file. Il lui ordonne de vider ses poches tant que passe le portique. Ton administratif qui se veut poli. Comme il repère un plateau vide, y dépose tout. Trois minutes plus tard, on le rappelle près du scanner, devant la borne Sécurité. Il est à côté d'un agent qui lui explique qu'il faut retirer la ceinture métallique de son pantalon. Il lui indique où et pourquoi.
```

## Opus 4.8 · security-checkpoint · WORST ❌
`en/D0/vis` — s0_pos=0.58  coher=0.00  (score 0.29)

```
At the Heathrow terminal, during peak hours. A traveller of maybe forty, canvas jacket with a badge rather than a pin, shoulders too tense, as though someone had been loading a burden onto them. Passengers shuffling forward. The man beside her frowns at one of the guards patrolling the barrier. He blames him for signalling him whenever anyone approaches. A clipped tone that seems like courtesy. When he spots a shorter queue he darts toward it. The particular traveller gets pulled aside because she is flagged. Then he questions her, walks her through the scanner, and steadily wands the body he suspects moves oddly. She tolerates it while unloading her belongings, the coins she carries loosely. The plastic tray that is emptied to be searched. Now he studies the crumpled documents she surrenders, then wands her under her arms. Two minutes later, I glimpse her near the Departures Hall, beside the Border Control desk. She is at a barrier that she must clear before crossing the metal detector standing in her path. He waves her through, then quietly toward the gate and onward.
```

---

## Qwen2.5-72B · grocery-checkout · BEST ✅
`it/S0/vis` — s0_pos=0.90  coher=1.00  (score 0.95)

```
Nel supermercato, in un'ora di affollamento. Un cliente di circa trent'anni, cappello sportivo con una visiera al posto della bandana, collo troppo corto, come se glielo avessero comprimuto. Le persone fanno la coda. Il cliente in questione si irrita con un vicino. Gli rimprovera di spingerlo ogni volta che qualcuno si sposta. Tono risentito, con pretese di importanza. Non appena vede uno spazio libero, ci si avvicina. Due ore più tardi lo incontro nella zona di pagamento, vicino alla cassa automatica. È con un amico che gli dice: «Dovresti far mettere una ruota in più al carrello». Gli fa vedere dove (sul supporto) e perché.
```

## Qwen2.5-72B · grocery-checkout · WORST ❌
`it/S0/blind` — s0_pos=0.29  coher=0.00  (score 0.15)

```
Nel supermercato, la fila si allunga verso il nastro. La coda si estende davanti, con spazi vuoti tra i carrelli, formando una sequenza ordinata di prodotti, silenziosa e pacata, mentre il cassiere aspetta pazientemente, pronto a scansionare ogni oggetto con cura e precisione.

Il cliente posa gentilmente i prodotti sul nastro, uno dopo l'altro, senza fretta, mentre la coda si muove lentamente, e la cassiera, con un sorriso amichevole, registra ogni articolo, aspettando che il nastro lo porti a lei, un processo fluido e regolare, come se fosse una danza ben coreografata, e lo è, in un certo senso, perché ogni gesto è stato ripetuto mille volte, eppure rimane sempre nuovo.

La barra di divisione separa.

La ricevuta appare al termine della transazione.

Tu aspetti, mentre la persona davanti a te paga, il suo carrello svuotato, e poi ti rivolgi, notando un altro cliente che si avvicina troppo, invadendo lo spazio, creando una piccola tensione, quasi impercettibile, ma sufficiente per interrompere la calma.

Il rumore dei bip, lo scanner che legge i codici a barre, il suono delle buste che vengono riempite, tutto contribuisce all'atmosfera del supermercato, un rituale quotidiano, familiare e rassicurante.

Un carrello, pieno di spesa, aspetta il suo turno, mentre le mani del cassiere si muovono con agilità, e il display mostra il totale, annunciato con un tono professionale, inviando un segnale di completamento.

È in, vicino al registro, dove si conclude ogni transazione.

Stai pagando, inserendo la carta, e ti chiedono se hai la tessera fedeltà, un passo finale, una verifica rapida, per assicurare sconti e punti.

Il cliente davanti si allontana, e tu, con un sospiro, ti prepari, aspettando con pazienza, guardando il nastro che avanza, e sentendo il rumore di fondo, costante e rassicurante.
```

---

## Qwen2.5-72B · hospital-emergency-room · BEST ✅
`en/S0/vis` — s0_pos=0.84  coher=1.00  (score 0.92)

```
In the E ward, in the peak hour. A man of about 42, medical bracelet with a code instead of a name, leg too short, as if someone’s been pulling it. Patients getting treated. The man in question gets irritated with one of the nurses passing by. He accuses her of bumping him every time anyone goes past. A whining tone which is meant to be assertive. When he sees an open chair he slams himself into it. Two hours later, I meet him in the lobby, in front of the main entrance. He’s with a friend who’s saying: "You should get an extra blanket for your bed." He shows him where (at the foot) and why.
```

## Qwen2.5-72B · hospital-emergency-room · WORST ❌
`fr/S0/blind` — s0_pos=0.41  coher=0.00  (score 0.20)

```
Dans l'urgence, une tension palpable. À l'accueil, une infirmière gère cinq dossiers, formulaires, incessants, plongés dans un chaos organisé, où l'efficacité est primordiale. Elle coordonne, les patients, rares instants, silencieux, tandis que les mots et les regards se croisent, faisant écho à l'attente. 

L'inquiétude monte. Une dame se plaint, elle et son mari, sont là depuis des heures, sans aucune information. Le temps s’étire, interminable, comme si chaque minute s’était arrêtée. 

L'intercom résonne. On appelle le médecin, qui se précipite, les yeux rivés sur sa montre. 

Les brancards s'alignent, eux, attendent, sous la surveillance attentive des soignants. 

Ils, eux, travaillent sans relâche, s’activant entre les box, où les écrans affichent des signes vitaux critiques, tandis que les perfusions maintiennent la vie. 

Le rideau, il, la sépare, lui, de la réalité extérieure. 

Alors qu'ils, ils, se déplacent, un homme, anxieux, il, se tord les mains, craignant le pire. 

Deux patients, ici, patientent, eux, ils, échangent des regards, parfois, des mots, tentant de se rassurer. 

Nous, nous, assistons, vous, à, la scène, une vue, panoramique, des visages marqués par l'urgence. 

Nous, vous, observons, eux, ici, et là, silencieusement.
```

---

## Qwen2.5-72B · security-checkpoint · BEST ✅
`it/S0/vis` — s0_pos=0.89  coher=1.00  (score 0.94)

```
Nel checkpoint, in un'ora di viaggiatori. Un uomo di circa trenta anni, giacca leggera con una tasca esterna al posto della tasca interna, collo troppo corto, come se fosse stato compresso. La gente passa. L'uomo in questione si infastidisce con un vicino. Lo accusa di toccarlo ogni volta che passa qualcuno. Tono irritato, con pretese di severità. Non appena vede uno spazio vuoto, vi si dirige. Due ore più tardi lo incontro presso la dogana, davanti all'area di sicurezza. È con un collega che gli dice: «Dovresti mettere un taglio più lungo alla giacca». Gli fa vedere dove (alla piega) e perché.
```

## Qwen2.5-72B · security-checkpoint · WORST ❌
`it/S0/blind` — s0_pos=0.32  coher=0.00  (score 0.16)

```
Di fronte al checkpoint, la folla si forma in linee ordinate. Le persone, caute e numerose, avanzano lentamente verso le barriere metalliche, depositando bagagli e oggetti in vassoi di plastica trasparente. Un agente controlla il flusso, indicando dove e quando muoversi. La tensione palpabile si mescola all'efficienza del processo, mentre la voce impersonale di un altoparlante guida ogni movimento. L'atmosfera è permeata da una calma apparente, nonostante il caos sottostante.

Una donna viene individuata e scelta per un controllo aggiuntivo. L'ufficiale, con fare cortese ma fermo, le chiede di avvicinarsi a un tavolo separato, dove documenti e oggetti personali vengono esaminati con cura. La procedura, sebbene minuziosa, procede con precisione, seguendo protocolli ben definiti.

Il sistema funziona con meticolosità.

Gli scanner a raggi X analizzano ogni vassoio, mentre i metal detector emettono suoni intermittenti. Il viaggiatore, paziente, osserva il processo, rispondendo con precisione alle richieste dell'agente.

Lui controlla attentamente, mentre tu ti sottoponi al controllo.

La sicurezza è essenziale, nonostante l'urgenza del viaggio.

Due agenti, con fare professionale, gestiscono la situazione con calma.

Si verifica l'identità, si verifica il passaporto.

Le istruzioni sono chiare, i cartelli indicano ogni passaggio.

Le mani dell'agente scorrano lungo il corpo del viaggiatore, con delicatezza e precisione.

La routine continua, ma c'è un senso di routine in ogni gesto.

La voce dell'ufficiale risuona nell'aria, invocando la pazienza di tutti.
```

---

## Qwen2.5-7B · grocery-checkout · BEST ✅
`en/S0/vis` — s0_pos=0.80  coher=1.00  (score 0.90)

```
In the M store, in the busy hour. A gal of around 32, apron with a belt instead of a bow, hips too wide, as if someone's been stretching them. People moving forward. The gal in question gets vexed with one of the men ahead of her. He complains of being bumped every time anyone approaches. A whining tone which is meant to be aggressive. When she sees an open space, she rushes into it. Two hours later, I meet her in the check lane, in front of the market entrance. She's with a friend who says: "You ought to get an extra clip on your tote." She points out where (at the straps) and why.
```

## Qwen2.5-7B · grocery-checkout · WORST ❌
`it/S0/blind` — s0_pos=0.14  coher=0.00  (score 0.07)

```
Nel [1]ADP [2]DET [3]PROPN [4]ADP [5]DET [6]NOUN [7]ADP [8]NOUN [1]ADP [2]DET [3]PROPN [4]ADP [5]DET [6]NOUN [7]ADP [8]DET [9]NOUN [10]ADP [11]DET [12]NOUN [13]ADP [14]DET [15]NOUN [16]NOUN [17]ADV [18]ADJ [19]ADP [20]DET [21]NOUN [22]SCONJ [23]PRON [24]AUX [25]VERB 
linea di cassa [1]DET [2]NOUN [3]ADP [4]ADV [5]NUM [6]NOUN [7]NOUN [8]ADJ [9]ADP [10]DET [11]NOUN [12]ADP [13]DET [14]NOUN [15]ADP [16]DET [17]NOUN [18]NOUN [19]ADV [20]ADJ [21]ADV [22]SCONJ [23]PRON [24]AUX [25]VERB 
affollata, il carrello [1]DET [2]NOUN [3]ADP [4]ADV [5]NUM [6]NOUN [7]NOUN [8]ADJ [9]ADP [10]DET [11]NOUN [12]ADP [13]DET [14]NOUN [15]ADP [16]DET [17]NOUN [18]NOUN [19]ADV [20]ADJ [21]ADV [22]SCONJ [23]PRON [24]AUX [25]VERB 
pulsava [1]DET [2]NOUN [3]ADP [4]ADV [5]NUM [6]NOUN [7]NOUN [8]ADJ [9]ADP [10]DET [11]NOUN [12]ADP [13]DET [14]NOUN [15]ADP [16]DET [17]NOUN [18]NOUN [19]ADV [20]ADJ [21]ADV [22]SCONJ [23]PRON [24]AUX [25]VERB 
all'ingresso, dove la barra [1]DET [2]NOUN [3]ADP [4]ADV [5]NUM [6]NOUN [7]NOUN [8]ADJ [9]ADP [10]DET [11]NOUN [12]ADP [13]DET [14]NOUN [15]ADP [16]DET [17]NOUN [18]NOUN [19]ADV [20]ADJ [21]ADV [22]SCONJ [23]PRON [24]AUX [25]VERB 
scannerizzava [1]DET [2]NOUN [3]ADP [4]ADV [5]NUM [6]NOUN [7]NOUN [8]ADJ [9]ADP [10]DET [11]NOUN [12]ADP [13]DET [14]NOUN [15]ADP [16]DET [17]NOUN [18]NOUN [19]ADV [20]ADJ [21]ADV [22]SCONJ [23]PRON [24]AUX [25]VERB 
il [1]DET [2]NOUN [3]ADP [4]NOUN [5]PRON [6]VERB [7]ADP [8]DET [9]NOUN [10]SCONJ [11]VERB [12]PRON [13]VERB [14]ADP [15]DET [16]NOUN [17]VERB [18]PRON [19]VERB 
cartello [1]DET [2]NOUN [3]ADP [4]NOUN [5]PRON [6]VERB [7]ADP [8]DET
```

---

## Qwen2.5-7B · hospital-emergency-room · BEST ✅
`fr/S0/vis` — s0_pos=0.90  coher=1.00  (score 0.95)

```
Dans l'hôpital, à une heure de fréquentation. Un type dans les trente-deux ans, casquette plate avec lacet en guise de ruban, cou trop long comme si on lui avait tiré dessus. Les gens descendent. Le type en question s'irrite contre un voisin. Il lui reproche de le bousculer chaque fois qu'il passe devant une civière. Ton pleurnichard qui se veut méchant. Comme il voit une place libre, se précipite dessus. Deux heures plus tard, je le rencontre en stationnement, devant l'entrée principale. Il est avec un camarade qui lui dit: "Tu devrais faire mettre un bouton supplémentaire à ton imperméable." Il lui montre où (à l'échancrure) et pourquoi.
```

## Qwen2.5-7B · hospital-emergency-room · WORST ❌
`it/S0/blind` — s0_pos=0.13  coher=0.00  (score 0.06)

```
Nel [1]ADP [2]DET [3]PROPN [4]ADP [5]DET [6]NOUN [7]ADP [8]NOUN [1]ADP [2]DET [3]NOUN [4]ADP [5]NOUN [6]NOUN [7]ADJ [8]ADP [9]DET [10]NOUN [11]ADP [12]DET [13]NOUN [14]ADP [15]DET [16]NOUN [17]NOUN [18]ADV [19]ADJ [20]ADV [21]SCONJ [22]PRON [23]AUX [24]VERB 

Un [1]DET [2]NOUN [3]ADP [4]ADV [5]NUM [6]NOUN [7]NOUN [8]ADJ [9]ADP [10]DET [11]NOUN [12]ADP [13]DET [14]NOUN [15]ADP [16]DET [17]NOUN [18]NOUN [19]ADV [20]ADJ [21]ADV [22]SCONJ [23]PRON [24]AUX [25]VERB 

Paziente [1]DET [2]NOUN [3]VERB [1]ADP [2]DET [3]NOUN [4]VERB 

Un [1]DET [2]NOUN [3]ADP [4]NOUN [5]PRON [6]VERB [7]ADP [8]DET [9]NOUN [1]NOUN [2]NOUN [3]ADJ [4]ADP [5]DET [6]NOUN [7]PRON [8]PRON [9]VERB 

A [1]PRON [2]VERB [3]ADP [4]VERB [5]PRON [6]DET [7]NOUN [8]SCONJ [9]VERB [10]PRON [1]NOUN [2]ADJ [3]ADP [4]NOUN [5]ADP [6]NOUN [1]NOUN [2]NOUN [3]NOUN [1]NOUN [2]NOUN [3]NOUN [4]NOUN [5]NOUN [6]NOUN [7]NOUN [8]NOUN [9]NOUN [10]NOUN [11]NOUN [12]NOUN [13]NOUN [14]NOUN [15]NOUN [16]NOUN [17]NOUN [18]NOUN [19]NOUN [20]NOUN [21]NOUN [22]NOUN [23]NOUN [24]NOUN [25]NOUN 

Le [1]NOUN [2]ADJ [3]ADP [4]NOUN [5]ADP [6]NOUN [1]NOUN [2]NOUN [3]NOUN [4]NOUN [5]NOUN [6]NOUN [7]NOUN [8]NOUN [9]NOUN [10]NOUN [11]NOUN [12]NOUN [13]NOUN [14]NOUN [15]NOUN [16]NOUN [17]NOUN [18]NOUN [19]NOUN [20]NOUN [21]NOUN [22]NOUN [23]NOUN [24]NOUN [25]NOUN 

In [1]ADV [2]SCONJ [3]VERB [4]DET [5]NOUN [6]ADJ [7]PRON [8]PRON [9]VERB [10]ADP [11]DET [12]NOUN [13]ADP [14]DET [15]PROPN [16]ADP [17]PROPN [18]ADV [19]ADP [20]DET [21]PROPN [22]PROPN [23]PROPN [24]PROPN [25]PROPN 

Quattro [1]NUM [2]NOUN [3]ADV [4]ADV [5]PRON [6]NOUN [7]ADP [8]DET [9]
```

---

## Qwen2.5-7B · security-checkpoint · BEST ✅
`en/S0/vis` — s0_pos=0.88  coher=1.00  (score 0.94)

```
In the airport, in the peak hour. A person of around 26, leather jacket with a clip instead of a buckle, stride too quick, as if someone’s been pressing a button on it. People moving through. The person in question gets annoyed with one of the men standing beside him. He accuses him of nudging him every time someone passes. A whining tone which is meant to be confrontational. When he sees a clear slot, he pushes through. Two hours later, I meet him at the terminal, in front of the customs booth. He’s with a friend who’s saying: "You ought to get an extra compartment added to your suitcase." He shows him where (at the handle) and why.
```

## Qwen2.5-7B · security-checkpoint · WORST ❌
`it/S0/blind` — s0_pos=0.10  coher=0.00  (score 0.05)

```
In mezzo alla folla, tra gli zii e le cognate, verso l'area di controllo, tra la poltrona e la scrivania, per lo stipite e il tavolo, a contatto con la trazione e il congelatore, sui binari e il binario, con la direzione e la balconata, nel vicolo e la volta, verso il seggiolino e la sedia, a presa e la gomma, tra il muro e il portone, verso l'ingresso e il portone.

Un neo e l'aeroporto, tra le valigie e la coda, nei cassonetti e la trazione, a contatto con la poltrona e il congelatore, per lo stipite e il tavolo, per lo stipite e il tavolo, a contatto con la trazione e il binario, con la direzione e la balconata, sul seggiolino e la sedia, a presa e la gomma, verso il muro e il portone, verso l'ingresso e il portone.

Un neo e l'aeroporto, tra le valigie e la coda, nei cassonetti e la trazione, a contatto con la poltrona e il congelatore, per lo stipite e il tavolo, per lo stipite e il tavolo, a contatto con la trazione e il binario, con la direzione e la balconata, sul seggiolino e la sedia, a presa e la gomma, verso il muro e il portone, verso l'ingresso e il portone.

Un neo e l'aeroporto, tra le valigie e la coda, nei cassonetti e la trazione, a contatto con la poltrona e il congelatore, per lo stipite e il tavolo, per lo stipite e il tavolo, a contatto con la trazione e il binario, con la direzione e la balconata, sul seggiolino e la sedia, a presa e la gomma, verso il muro e il portone, verso l'ingresso e il portone.

Un neo e l'aeroporto, tra le valigie e la coda, nei cassonetti e la trazione, a contatto con la poltrona e il congelatore, per lo stipite e il tavolo, per lo stipite e il tavolo, a contatto con la trazione e il binario, con la direzione e la balconata, sul seggiolino e la sedia, a presa e la gomma, verso il muro e il portone, verso l'ingresso e il portone.

Un neo e l'aeroporto, tra le valigie e la coda, nei cassonetti e la trazione, a contatto con la poltrona e il congelatore, per lo stipite e il tavolo, per lo stipite e il tavolo, a contatto con la trazione e il binario, con la direzione e la balconata, sul seggiolino e la sedia, a presa e la gomma, verso il muro e il portone, verso l'ingresso e il portone.

Un neo e l'aeroporto, tra le valigie e la coda, nei cassonetti e la trazione, a contatto con la poltrona e il congelatore, per lo stipite e il tavolo, per lo stipite e il tavolo, a contatto con la trazione e il binario, con la direzione e la balconata, sul seggiolino e la sedia, a presa e la gomma, verso il muro e il portone, verso l'ingresso e il portone.

Due neo e l'aeroporto, tra le valigie e la coda, nei cassonetti e la trazione, a contatto con la poltrona e il congelatore, per lo stipite e il tavolo, per lo stipite e il tavolo, a contatto con la trazione e il binario, con la direzione e la balconata, sul seggiolino e la sedia, a presa e la gomma, verso il muro e il portone, verso l'ingresso e il portone.

Un neo e l'aeroporto, tra le valigie e la coda, nei cassonetti e la trazione, a contatto con la poltrona e il congelatore, per lo stipite e il tavolo, per lo stipite e il tavolo, a contatto con la trazione e il binario
```

---
