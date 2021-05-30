# Relazione del primo gruppo di esercizi

### TLN_dicaro_1.1

_CONSEGNA_:

* Date delle definizioni per quattro concetti (due concreti e due astratti),
calcolare la similarità fra di esse.
  

* Aggregare anche le definizioni secondo le dimensioni di concretezza e
specificità e ri-calcolare i punteggi.
  
  
* Effettuare del pre-processing se necessario prima del calcolo.

  
_SVOLGIMENTO_:

**I termini sono**:

```
                    Generico    Specifico
                ==========================
    Concreto    |   Paper     Sharpener
    Astratto    |   Courage   Apprehension
```

* Si è scelto di filtrare le stopwords come fase di pre-processing, per
concentrarsi sui termini salienti.
    
* Abbiamo provato, oltre la baseline, come misura di similarità la Cosine Similarity 
andando ad aggregare i risultati

_RISULTATI_:

![1.1 - plot_1](output/aggregate/Baseline.png "Baseline")

* Si è notato come, nel caso di termini concreti, la similarità sia
significativamente più elevata di quanto non acccada per i termini astratti.
Questo è probabilmente dovuto alla possibilità di utilizzare degli attributi
visivi per descrivere il termine.

* Nel caso dei termini astratti, invece, la mancanza di questi attributi
concreti porta a definizioni meno simili fra di loro.

![1.1 - plot_2](output/aggregate/Cosine_Similarity_Experiment.png "Cosine Similarity")

### TLN_dicaro_1.2

_CONSEGNA_: 

* Dare una spiegazione dei risultati ottenuti nell’esercizio precedente

_SVOLGIMENTO_:

* Ho scelto di utilizzare approccio basato sul pos tagging delle parole

* Calcola la sovrapposizione tra i due set di definizioni pre-elaborate convertite in POS tagging.

_RISULTATI_:

![1.1 - plot_3](output/aggregate/POS_Experiment.png "POS_Experiment")

* Quello che si può evincere sembra essere una maggiore rilevanza statistica delle parole per i termini generici rispetto a quelli specifici e di quelli astratti rispetto a quelli concreti.

* Questo sembra confermare che i termini generici tendono ad essere meno omogenei nelle definizioni 
mentre i termini specifici utilizzano più spesso le stesse parole:

* In particolare l'oggetto concreto e specifico (sharpener) che ha la maggiore similarità ha un valore di POS nella media, rivelando che le definizioni sono molto simili dal punto di vista del Part to Speach, avendo gli stessi termini più o meno distribuiti in tutte le definizioni.

### TLN_dicaro_1.3

_CONSEGNA_:

* Implementazione di un semplice sistema di WSI e della pseudo-word evaluation

_SVOLGIMENTO_:

* Abbiamo usato uno spazio pre-costruito in gensin (glove-wiki-gigaword-300); questo serve a darci un embedding per una certa parola dati tutti i suoi possibili contesti.
* Abbiamo usato il brown corpus per scegliere una parola con frequenza tra 50 and 1000. Abbiamo scelto “bar” e “special” come parole di interesse (con frequenza rispettivamente 71 e 233).
* Abbiamo creato un contesto tenendo conto anche delle stop words e un embedding per la parola di interesse.
* Successivamente abbiamo applicato K-Means come algoritmo di cluster non supervisionato per predire i raggruppamenti.Ad esempio nel caso della parola “special” abbiamo scelto 10 clusters basandoci sul numero di synsets di WordNet.
* Possiamo vedere la distribuzione delle parole nei vari clusters
  
_RISULTATI_:

#### Embeddings e wordnet

![special_10](https://user-images.githubusercontent.com/37343600/120099635-e3dc5b80-c13c-11eb-860a-92acd808eb91.png)

* Abbiamo provato a fare un’interpretazione di questi clusters basandoci sui synsets di WordNet e usando l’algoritmo di Lesk. In particolare applicando Lesk a tutte le frasi vengono effettivamente identificati i 10 possibili sensi

![synset_lesk](https://user-images.githubusercontent.com/37343600/120099663-04a4b100-c13d-11eb-820b-e56043f27dce.png)

* Non vi è tuttavia una concordanza tra i cluster identificati con K-Means e i WordNet synsets.
* Questo significa che K-Means non riesce a separare i contesti nello stesso modo di un essere umano poiché i vari clusters non sono associabili ad uno specifico synset di WordNet;
* ogni cluster contiene frasi che possono essere assegnate a più synsets.
* Se ad esempio andiamo a vedere la distribuzione dei synsets per i primi quattro clusters (index 0, 1, 2, 3) possiamo vedere questa incongruenza:

![distribuzione_syns](https://user-images.githubusercontent.com/37343600/120099701-3b7ac700-c13d-11eb-9798-b6f2717e58b2.png)

#### Pseud-words

* Infine abbiamo utilizzato il meccanismo a pseudo-word per vedere se il K-Means riusciva a separare i due contesti.
* Abbiamo perciò unito le parole in “barspecial” e rifatto la stessa analisi applicando poi K-Means con 2 clusters.
* I risultati questa volta sono più incoraggianti:
  + il cluster 0 ha 55 matching per “bar” e 55 matching per “special”
  + il cluster 1 ha 13 matching per “bar” e 165 matching per “special”
* Anche se non perfettamente si può vedere che K-Means riesce ad associare i diversi contesti in maniera abbastanza precisa.
* In particolare, si può vedere dalla disomogeneità del raggruppamento, che il cluster 0 è più associato al contesto di bar mentre il cluster 1 a quello di special.
* Abbiamo anche calcolato le misure statistiche Precision, Recall e F1-score:

| Metric | cluster 0 | cluster 1 |
| ------------- | ------------- |  ------------- |
| Precision    |    0.50      |   0.93 |
| Recall       |    0.80      |   0.73 |
| F1-score     |    0.62      |   0.82 |



### TLN_dicaro_1.4

_CONSEGNA_:
* Implementare un sistema basato sulla teoria di Hanks per la costruzione del
significato.
  

* Scelto un verbo transitivo (quindi valenza >= 2), recuperare da un corpus
delle istanze in cui viene usato.
  

* Effettuare il parsing di queste frasi per identificare i supersensi di
WordNet associati agli argomenti del verbo (subject e object).
  

* Calcolare le frequenze di questi supersensi per i due ruoli e stampare le
possibili combinazioni.
  

_SVOLGIMENTO_:
* Si è scelto il verbo 'to break', in particolare il presente terza persona singolare.


* Il corpus utilizzato è Wikipedia, da cui sono state estratte 3000 frasi, usando 
  [sketch engine](https://www.sketchengine.eu/)


* Per il parsing a dipendenze si è usata la libreria [spaCy](https://spacy.io/).


* Sono state scartate quelle frasi in cui il verbo non presenta entrambi i ruoli richiesti.
  

* I termini che svolgono i ruoli vengono lemmatizzati e si va poi a calcolare
  il loro synset migliore tramite WSD (algoritmo di Lesk).


* Nel caso il soggetto sia 'he'/'she', è necessario forzare il suo synset a 'person.n.01' per evitare che
  venga erroneamente riconosciuto come 'elio', analogamente per 'it' sostituito ad 'artifact.n.01'


* Con questi synset si individua il relativo supersenso `lexname`, andando a calcolare
  poi frequenze e combinazioni possibili.


* Si creano poi due grafici con le migliori `k` coppie soggetto-oggetto, usando due versioni 
  differenti dell' algoritmo Lesk
  
_RISULTATI_:

![plot_1.4.1](output/hanks/Our_Lesk_for_breaks.png "Our Lesk" )


![plot_1.4.2](output/hanks/NLTK_Lesk_for_breaks.png "NLTK Lesk" )

## TLN_dicaro_1.5

_CONSEGNA_:
* Esperimento content-to-form usando i dati dell’esercizio 1.1
* Per ogni concetto, prendere le definizioni a disposizione, cercare in WordNet il synset corretto utilizzando il principio del “genus” per indirizzare la ricerca

_SVOLGIMENTO_:

* Abbiamo inizialmente verificato che ognuno dei quattro termini avesse almeno un corrispettivo synset su WordNet avendo riscontro positivo.
* Applicando il principio del genus per ogni definizione abbiamo estratto le prime n = 4 parole significative rifacendomi alla teoria che il significato sia maggiormente racchiuso all’inizio della definizione.
* Per ogni parola significativa abbiamo ricercato i rispettivi iponimi.
* Abbiamo inferito il WordNet synset in base alla massima similarità tra le nostre definizioni e quelle associate alle definizioni estratte dagli iponimi

_RISULTATI_:

* Non in tutti i casi, purtroppo, l’algoritmo è riuscito ad inferire il senso corretto della parola; in alcuni casi abbiamo verificato che il senso corretto non appariva all’interno degli iponimi estratti

* Questi sono i risultati complesssivi ottenuto in termini di inferenza e massima similarità ottenuta:

| Concept | max similarity | synset inferred 
| ------------- | ------------- |  ------------- |
|  Paper  | 0.75           | paper.n.01 
| Courage | 0.75           | physical_ability.n.01 
| Apprehension |    1.00   | apprehension.n.01
| Sharpener |   0.50       | acuminate.v.01

* Come possiamo vedere solo in due casi (paper e apprehension), l’algoritmo riesce ad inferire correttamente il senso a partire dalle definizioni

* Ecco invece la lista dei top 5 sensi che l'algoritmo ci presenta

**COURAGE**

|n°  | Name: Courage, dtype: object|
| ------------- | ------------- |
| 0 |   [property, allows, face]
| 1 |    [ability, face, fears]
| 2 |     [ability, face, thing]
| 3 |   [inner, strength, thaht]
| 4 |   [ability, control, fear]

*The best word forms for concept are*

| Score | Synset |
| ------------- | ------------- |
 0.6666666666666666  |Synset('physical_ability.n.01') |
 0.5  |Synset('confront.v.04')|
 0.2857142857142857  |Synset('take_the_bull_by_the_horns.v.01')
 0.25  |Synset('lee.n.08')
 0  |Synset('countenance.n.03')|
 
 
 
```
max_sym is: 0.75
my wn synset inferred for courage is: Synset('physical_ability.n.01')
The best word forms for courage concept
Score: 0.6666666666666666 for synset: Synset('physical_ability.n.01')
Score: 0.5 for synset: Synset('confront.v.04')
Score: 0.2857142857142857 for synset: Synset('take_the_bull_by_the_horns.v.01')
Score: 0.25 for synset: Synset('lee.n.08')
Score: 0 for synset: Synset('countenance.n.03')
```

```
Computing concept paper
0          [cellulose, material, cut, folded, written]
1    [material, derived, trees, used, several, cont...
2                    [type, material, made, cellulose]
3        [product, obtained, wood, cellulose, ., used]
4          [flat, material, made, wood, used, writing]
Name: Paper, dtype: object
max_sym is: 0.6666666666666666
my wn synset inferred for paper is: Synset('coloring_material.n.01')
The best word forms for paper concept
Score: 0.5 for synset: Synset('coloring_material.n.01')
Score: 0.4444444444444444 for synset: Synset('animal_material.n.01')
Score: 0.4 for synset: Synset('aggregate.n.02')
Score: 0.3333333333333333 for synset: Synset('diethylaminoethyl_cellulose.n.01')
Score: 0.2857142857142857 for synset: Synset('carboxymethyl_cellulose.n.01')
Score: 0.25 for synset: Synset('carboxymethyl_cellulose.n.01')
Score: 0 for synset: Synset('carboxymethyl_cellulose.n.01')
```

```
 Computing concept apprehension
0    [something, strange, causes, strange, feeling,...
1                 [fearful, expectation, anticipation]
2                        [moode, one, feel, agitation]
3                                 [state, disturbance]
4                                      [worry, future]
Name: Apprehension, dtype: object
max_sym is: 1.0
my wn synset inferred for apprehension is: Synset('apprehension.n.01')
The best word forms for apprehension concept
Score: 0.8571428571428571 for synset: Synset('apprehension.n.01')
Score: 0.5714285714285714 for synset: Synset('expectation.n.03')
Score: 0.5 for synset: Synset('expectation.n.03')
Score: 0.4444444444444444 for synset: Synset('emotion.n.01')
Score: 0.4 for synset: Synset('astonishment.n.01')
Score: 0.3333333333333333 for synset: Synset('affection.n.01')
Score: 0.2857142857142857 for synset: Synset('affection.n.01')
Score: 0.25 for synset: Synset('affect.n.01')
Score: 0.2222222222222222 for synset: Synset('affect.n.01')
Score: 0.18181818181818182 for synset: Synset('affect.n.01')
Score: 0 for synset: Synset('affect.n.01')
```

```
 Computing concept sharpener
0    [tool, equipped, blade, allows, sharpen, tip]
1                   [object, used, shapen, pencil]
2                        [object, sharpen, pencil]
3                   [tool, used, sharpen, pencils]
4         [little, object, allow, sharpen, pencil]
Name: Sharpener, dtype: object
max_sym is: 0.7272727272727273
my wn synset inferred for sharpener is: Synset('acuminate.v.01')
The best word forms for sharpener concept
Score: 0.5454545454545454 for synset: Synset('acuminate.v.01')
Score: 0.5 for synset: Synset('drill.n.01')
Score: 0.4 for synset: Synset('cutting_implement.n.01')
Score: 0.2 for synset: Synset('abrader.n.01')
Score: 0.16666666666666666 for synset: Synset('abrader.n.01')
Score: 0 for synset: Synset('abrader.n.01')
```
