Preberi `AGENTS.md` in `MEMORY.md`, nato naredi analizo dejanskih HCI/MAHNOB podatkov z vsemi lokalno prisotnimi udeleženci.

Kontekst: zanima me, ali je trenutni GNN v2 način gradnje `fixation` povezav smiseln. Trenutna implementacija v `src/data/data.py` poveže samo zaporedna vozlišča z istim `fixation-index` v obe smeri. Ker imamo temporalne povezave s `kt>=1`, se ti robovi topološko prekrivajo s temporalnimi robovi. Razmišljam o alternativi: povezati vozlišča z istim `fixation-index` kot sparse clique, npr. full clique z 90% dropoutom oziroma keep probability 0.1, po možnosti resamplano vsako epoho. Ali druga alternativa: povežemo vozlišče i z vozlišči i+F/kf, i+2*F/kf, ..., i+kf*F/kf, kjer je F = število vozlišč v fiksaciji, katere del je vozlišče i, kf pa je koeficient gostote (npr. kf = 10). Tretja alternativa je da naredim full clique znotraj fiksacije - torej da povežem s fikacijskimi povezavami vsa vozlišča znotraj dane fiksacije. Za vse alternative poračunaj koliko povezav to nanese in kolikšen delež vseh povezav v grafu (v povprečju) bi to predstavljalo. Poračunaj tudi podatek, koliko imamo vseh fiksacij (povprečno) v grafu. 

Naloga:
1. Najdi dejanske lokalne HCI/MAHNOB podatke, ki vsebujejo `fixation-index`, in jasno poročaj, katere datoteke/subjekti/recordingi so bili uporabljeni. Uporabi podatke 4 naključnih subjektov.
2. Uporabi iste ključne nastavitve kot trenutni quick/Table-6 GNN v2 config, če obstajajo: 10s okna, `min_samples_per_window=60`, `kt=2`, `ks=2`, izključeni subjekti `P9`, `P12`, `P15`, in samo emotion-elicitation, če je stolpec `experiment-type` na voljo.
3. Za vse uporabne 10s grafe izračunaj statistiko:
   - število vozlišč na graf;
   - število vozlišč z veljavnim `fixation-index`;
   - število fixation skupin na graf;
   - velikosti fixation skupin;
   - trenutno število sekvenčnih directed same-fixation robov;
   - število full directed same-fixation clique robov, tj. za skupino velikosti `m`: `m * (m - 1)`;
   - pričakovano število robov pri keep probability 0.1;
   - dejansko/ocenjeno število temporalnih robov pri `kt=2`;
   - dejansko/ocenjeno število spatial robov pri `ks=2`, če je praktično;
   - delež temporalnih robov, ki povezujejo vozlišča z istim `fixation-index`;
   - delež trenutnih sekvenčnih fixation robov, ki so že temporalni robovi.
4. Poročaj aggregate statistiko po vseh grafih in po subjektih: count, mean, median, p90, p95, p99, max. Posebej izpostavi worst-case okna.
5. Kratko odgovori:
   - ali je full clique pregost;
   - ali je clique z 90% dropoutom približno izvedljiv;
   - ali je bolj smiselno dropout fiksirati pri gradnji grafa ali resamplati vsako epoho;
   - ali trenutni sekvenčni fixation pristop dodaja kaj več kot ločen relation label nad že obstoječimi temporalnimi robovi.
6. Če narediš skripto, naj bo začasna ali dobro poimenovana; ne pusti nepotrebnih testnih datotek. Na koncu napiši kratek report z najpomembnejšimi tabelami in konkretnim priporočilom za naslednji ablation.
