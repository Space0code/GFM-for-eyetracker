---

```md
# 2. Podrobni operativni načrt

## 0. Trenutne potrjene odločitve

- [x] Naslov: Grafovska nevronska mreža za prepoznavo čustev iz sledilnika pogleda
- [x] Jezik: slovenščina
- [x] Glavni dataset: MAHNOB-HCI-TAGGING
- [x] eSEEd_v2: samo zgodovinska motivacija
- [x] Vhod: samo eye-tracking signali
- [x] Glavni nalogi: 3-class arousal in 3-class valence
- [x] Split: subject LOO in recording LOO
- [x] Combined LOO: izpuščeno
- [x] GFM/foundation model: future work
- [x] GazeMAE: SOTA primerjava
- [x] Finalni GNN: SpatioTemporalHeteroGNN
- [x] Default okno: 10 s
- [x] 5 s okno: ablation
- [x] Vozlišče: en sample
- [x] Spatial edges: kNN v `(x, y)`
- [x] Temporal edges: radius `k_t`
- [x] Temporal forward/backward: ločena edge tipa
- [x] Edge weights: glavni del modela

---

# 1. Pisanje, ki ga lahko začneš takoj

## 1.1 Uvod

**Status:** lahko pišeš zdaj.

**TODO vsebina:**

- [ ] Kratek kontekst:
  - eye-tracking kot neinvaziven vir informacij o pozornosti in kognitivno-afektivnem stanju.
- [ ] Problem:
  - klasični pristopi pogosto agregirajo signal v ročno izdelane značilke,
  - s tem lahko izgubijo lokalno časovno/prostorsko strukturo pogleda.
- [ ] Predlagana ideja:
  - okno eye-tracking signala predstavimo kot graf.
- [ ] Raziskovalno vprašanje.
- [ ] Prispevki.

**Predlagani prispevki:**

- [ ] Konstrukcija spatio-temporalnih grafov iz eye-tracking oken.
- [ ] GNN arhitektura z ločeno obravnavo prostorskih in časovnih povezav.
- [ ] Primerjava z ne-grafovskimi modeli na enakih vhodnih oknih.
- [ ] Primerjava z GazeMAE + MLP kot predstavnikom self-supervised gaze reprezentacij.
- [ ] Ablacijska analiza vpliva pupil značilk, prostorskih povezav, časovnih povezav in uteži povezav.

---

## 1.2 Sorodna dela

**Status:** lahko pišeš zdaj, ampak reference še preveri pred oddajo.

**Sekcije:**

- [ ] Prepoznavanje čustev iz eye-trackinga.
- [ ] MAHNOB-HCI in primerjalni protokoli.
- [ ] Grafovske nevronske mreže.
- [ ] Spatio-temporalni grafi.
- [ ] Gaze representation learning.
- [ ] GazeMAE.
- [ ] Foundation models kot future work.

**Posebej pomembno:**

- MAHNOB-HCI paper je uporabil 3-class arousal in 3-class valence.
- Njihov eye-gaze baseline ni uporabljal samo surovih `(x, y, pupil)` signalov, temveč 38 handcrafted značilk.
- Njihov split je participant-independent leave-one-participant-out.
- Zato primerjava ne bo popolnoma ena-na-ena, ampak je dovolj motivacijska.

V journalu je že zapisano, da MAHNOB-HCI paper za ET baseline uporablja 3-class arousal/valence, keyword-derived labele, 38 handcrafted eye-gaze features, RBF SVM in participant-independent LOO.  [oai_citation:5‡journal.md](sediment://file_00000000ea5c72468d54b4e56627ab09)

---

## 1.3 Podatki

**Status:** lahko pišeš zdaj.

**TODO vsebina:**

- [ ] MAHNOB-HCI opis.
- [ ] Emotion elicitation del.
- [ ] Subjekti.
- [ ] Posnetki.
- [ ] Self-report oznake.
- [ ] Valence/arousal 9-točkovna skala ali keyword-derived mapiranje — to moraš še dokončno potrditi.
- [ ] Eye-tracking signali.
- [ ] Zakaj ne uporabljaš drugih modalnosti.

**Odprto vprašanje:**

- [ ] Ali boš 3-class valence/arousal tvoril:
  - iz numeric self-report ratings,
  - ali iz keyword feedback mappinga kot MAHNOB paper?

Če želiš najvišjo primerljivost z MAHNOB paperjem, uporabi keyword-derived mapping. Če želiš bolj neposredno uporabo self-report dimenzij, uporabi numeric ratings. To je pomembna metodološka odločitev.

---

## 1.4 Predobdelava

**Status:** lahko pišeš zdaj, če je pipeline stabilen.

**TODO vsebina:**

- [ ] Branje raw/processed HCI podatkov.
- [ ] Izbira emotion-elicitation sekcij.
- [ ] Odstranitev neoznačenih baseline/neutral period, če jih ne uporabljaš.
- [ ] Obdelava NaN.
- [ ] Outlier filtering za `(x, y)`.
- [ ] Outlier filtering za pupil.
- [ ] Normalizacija.
- [ ] Train/test leakage zaščita.

**Posebej pazi:**

- Normalizacija mora biti fold-safe.
- Če delaš per-subject normalizacijo, moraš jasno razložiti, kaj je znano v test času.
- Pri subject LOO je per-subject normalizacija test subjekta metodološko občutljiva, če uporablja statistike celotnega test subjekta.

**TODO odločitev:**

- [ ] Določiti finalno normalizacijo:
  - global train-fold normalization,
  - per-recording normalization,
  - per-subject normalization,
  - robust normalization.

---

## 1.5 Windowing

**Status:** lahko pišeš zdaj.

**Potrjeno:**

- default window length = 10 s,
- 5 s = ablation.

**TODO vsebina:**

- [ ] Zakaj okna.
- [ ] Kako se okna izrežejo.
- [ ] Ali se prekrivajo.
- [ ] Kateri label dobi okno.
- [ ] Kaj narediš z okni brez labela.
- [ ] Kako ravnaš z manjkajočimi vzorci znotraj okna.
- [ ] Koliko vzorcev približno vsebuje 10 s okno.
- [ ] Koliko grafov nastane.

**TODO odločitev:**

- [ ] Ali so okna non-overlapping ali sliding.
- [ ] Če sliding: stride.
- [ ] Če non-overlapping: kako obravnavaš zadnji krajši segment.

---

## 1.6 Grafovska konstrukcija

**Status:** lahko pišeš zdaj.

**Potrjeno:**

```text
node = one sample in window
node features = x, y, pupil-left, pupil-right
spatial edges = kNN in gaze coordinate space
temporal edges = radius kt
temporal edge types = forward and backward
edge weights = used