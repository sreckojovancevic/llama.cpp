\# Phase 2 — Dynamic MoE Expert Residency \& Queueing



\## Status



\*\*Design Review — No Code\*\*



Ovaj dokument definiše tehnički koncept za sledeću fazu WeightProvider-a.



U ovoj fazi nema implementacije. Cilj je da se potvrdi:



\* arhitektonska izvodljivost u llama.cpp/ggml execution modelu;

\* način sinhronizacije rezidencije;

\* način upravljanja VRAM slotovima;

\* ponašanje pri cache miss-u;

\* admission politika za promociju;

\* korist temporal locality pristupa;

\* mogućnost kasnijeg uvođenja prediction/prefetch mehanizma.



Implementacija se ne započinje pre završetka design review-a i analize rezultata Phase 1.



\---



\# 1. Existing Phase 1



Phase 1 uvodi statički WeightProvider za MoE eksperte.



Postojeći model:



```text

&#x20;                   MoE Router

&#x20;                       │

&#x20;                       ▼

&#x20;                Expert selection

&#x20;                       │

&#x20;             ┌─────────┴─────────┐

&#x20;             ▼                   ▼

&#x20;           HOT                 COLD

&#x20;             │                   │

&#x20;             ▼                   ▼

&#x20;           VRAM              CPU\_Mapped

&#x20;             │                   │

&#x20;             ▼                   ▼

&#x20;           GPU                 CPU

&#x20;             │                   │

&#x20;             └─────────┬─────────┘

&#x20;                       ▼

&#x20;                     Merge

```



Phase 1 je funkcionalno potvrđen kroz:



\* statički HOT/COLD placement;

\* CPU execution cold eksperata;

\* GPU execution hot eksperata;

\* routing → residency → compute → merge;

\* exact correctness testove;

\* CPU/RPC bit-level proveru;

\* CUDA numeričku proveru;

\* VRAM placement reporting.



Važna karakteristika Phase 1 je da \*\*CPU execution već predstavlja funkcionalni fallback za eksperte koji nisu u VRAM-u\*\*.



Phase 2 koristi taj mehanizam kao osnovu za dinamičku rezidenciju.



\---



\# 2. Phase 2 Objective



Cilj Phase 2 je pretvoriti statički placement u:



> \*\*dynamic expert residency system\*\*



VRAM postaje dinamički cache za aktivne MoE eksperte.



CPU-mapped memorija ostaje fallback i veći residency nivo.



Osnovni princip:



```text

Router

&#x20; │

&#x20; │ WHAT?

&#x20; ▼

Expert request

&#x20; │

&#x20; ▼

Residency Manager

&#x20; │

&#x20; │ WHERE?

&#x20; ▼

VRAM / CPU

&#x20; │

&#x20; ▼

Scheduler

&#x20; │

&#x20; │ WHEN / HOW?

&#x20; ▼

Compute

```



\---



\# 3. Fundamental Performance Rule



GPU \*\*nikada ne sme biti blokiran čekanjem na promotion eksperta\*\*.



Za svaki expert request:



```text

VRAM HIT

&#x20;   │

&#x20;   ▼

GPU compute

```



ili:



```text

VRAM MISS

&#x20;   │

&#x20;   ├──────────────► CPU compute immediately

&#x20;   │

&#x20;   └──────────────► background promotion candidate

```



Ne dozvoljava se:



```text

VRAM MISS

&#x20;   │

&#x20;   ▼

PCIe transfer

&#x20;   │

&#x20;   ▼

WAIT

&#x20;   │

&#x20;   ▼

GPU compute

```



Razlog je jednostavan:



Cold expert već postoji u CPU-mapped memoriji i može odmah biti izvršen na CPU-u.



Promotion je \*\*performance optimization\*\*, a ne correctness dependency.



\---



\# 4. Phase 2 Development Strategy



Phase 2 se deli na dve podfaze.



\## Phase 2a — Dynamic Residency Between Tokens



Prva implementacija ne koristi intra-token prediction.



Fokus:



\* dynamic VRAM residency;

\* fixed VRAM slots;

\* residency table;

\* temporal locality;

\* LRU ili slična osnovna politika;

\* promotion admission;

\* eviction;

\* između-tokena commit;

\* CPU fallback;

\* queueing;

\* correctness i performance measurement.



```text

Phase 2a



Token N

&#x20;  │

&#x20;  ▼

existing residency

&#x20;  │

&#x20;  ▼

compute

&#x20;  │

&#x20;  ▼

collect usage

&#x20;  │

&#x20;  ▼

background transfers

&#x20;  │

&#x20;  ▼

TOKEN BOUNDARY

&#x20;  │

&#x20;  ▼

commit completed changes

&#x20;  │

&#x20;  ▼

Token N+1

```



Ova faza treba da pokaže da sama temporal locality daje dovoljno koristi da opravda kompleksnost sistema.



\---



\## Phase 2b — Predictor / Intra-token Prefetch



Tek nakon validacije Phase 2a razmatra se:



\* predictor;

\* prediction confidence;

\* hidden-state-based prediction;

\* intra-token prefetch;

\* layer-aware scheduling;

\* dodatna sinhronizacija između slojeva;

\* CUDA stream koordinacija tokom tokena.



Phase 2b nije preduslov za funkcionalnost sistema.



\---



\# 5. Why Prediction Is Not Phase 2a



Router sloja N odlučuje eksperte za sloj N.



Ne postoji garancija da su eksperti za sloj N+1 poznati pre izvršavanja tog routera.



Zato:



```text

Layer N

&#x20; │

&#x20; └── Router N

&#x20;         │

&#x20;         ▼

&#x20;      actual E

```



ne može direktno da garantuje:



```text

Layer N+1

&#x20; │

&#x20; └── unknown until router N+1

```



Prediction bi morao da koristi dodatnu informaciju, na primer hidden state prethodnog sloja:



```text

Hidden state N

&#x20;     │

&#x20;     ▼

&#x20;  Predictor

&#x20;     │

&#x20;     ▼

Predicted experts N+1

```



Prediction je zbog toga zaseban optimizacioni sloj.



Njegova greška ne sme uticati na correctness.



```text

prediction hit

&#x20;   → possible prefetch benefit



prediction miss

&#x20;   → normal CPU fallback

```



\---



\# 6. Temporal Locality First



Phase 2a koristi činjenicu da eksperti koji su skoro korišćeni mogu imati povećanu verovatnoću ponovne upotrebe.



Početni model:



```text

recently used expert

&#x20;       ↓

higher residency priority

```



LRU predstavlja početnu eviction politiku.



Postojeća simulacija pokazuje temporal-locality signal sa približno:



```text

82–89% LRU locality

```



na testiranom workload-u.



Ovaj rezultat treba tretirati kao \*\*motivaciju za merenje\*\*, a ne kao unapred garantovan production rezultat.



\---



\# 7. VRAM Slot Model



Dynamic residency ne sme zavisiti od proizvoljnog menjanja backing memory strukture unutar već izgrađenog ggml grafa.



VRAM se zato konceptualno organizuje kao skup fiksnih slotova.



```text

VRAM

┌────────────────────────────┐

│ SLOT 0                     │

├────────────────────────────┤

│ SLOT 1                     │

├────────────────────────────┤

│ SLOT 2                     │

├────────────────────────────┤

│ SLOT 3                     │

├────────────────────────────┤

│ ...                        │

└────────────────────────────┘

```



Expert nije trajno vezan za slot.



Residency table održava mapiranje:



```text

Expert → Residency state → Slot

```



Primer:



```text

Expert    Location        State          Slot

\------------------------------------------------

E0        CPU             RESIDENT       -

E1        VRAM            RESIDENT        2

E2        CPU             RESIDENT       -

E3        VRAM            RESIDENT        0

E4        CPU             RESIDENT       -

E5        VRAM            PROMOTING       3

E6        VRAM            RESIDENT        1

```



\---



\# 8. Residency Table



Residency table predstavlja authoritative metadata sloj.



Minimalni podaci:



```text

Expert ID

Location

State

VRAM slot

last\_use

use\_count

pending operation

generation

```



Conceptual example:



```text

E5:

&#x20;   location = CPU

&#x20;   state = PROMOTING

&#x20;   slot = 3

&#x20;   pending = PROMOTION

```



Tabela se ne sme ažurirati tako da prijavi expert kao GPU-resident pre nego što je njegov transfer završen i potvrđen.



\---



\# 9. Residency Consistency



Ovo je fundamentalno pravilo sistema.



\*\*Residency changes become visible only at a token boundary.\*\*



Tokom tokena `t`:



```text

┌──────────────────────────────────────┐

│ Residency table = stable             │

│                                      │

│ GPU may read current slots           │

│ Background transfers may execute     │

└──────────────────────────────────────┘

```



Background transfer može biti pokrenut tokom tokena.



Na primer:



```text

Token t



GPU:

&#x20;   compute



Transfer stream:

&#x20;   CPU → VRAM

```



Ali rezultat transfera nije odmah logički vidljiv compute sistemu.



Na kraju tokena:



```text

CUDA event

&#x20;   │

&#x20;   ▼

transfer completed?

&#x20;   │

&#x20;   ▼

TOKEN BOUNDARY

&#x20;   │

&#x20;   ▼

commit residency changes

```



Samo završeni transferi mogu biti committed.



\---



\# 10. Slot Reuse Safety



Slot ne sme biti prepisan dok GPU još može da koristi prethodni sadržaj.



Zato eviction/reuse zahteva dokaz da je prethodni GPU usage završen.



Conceptual flow:



```text

Slot 3 → E5

&#x20;  │

&#x20;  ▼

E5 used by GPU

&#x20;  │

&#x20;  ▼

GPU completion event

&#x20;  │

&#x20;  ▼

slot safe to reuse

&#x20;  │

&#x20;  ▼

E7 → Slot 3

```



Nikada:



```text

GPU reading E5

&#x20;      │

&#x20;      └──► overwrite Slot 3 with E7

```



Ovo pravilo je neophodno za:



\* correctness;

\* reproducibility;

\* izbegavanje race condition-a;

\* izbegavanje use-after-overwrite situacija.



\---



\# 11. Request Queue



Queue predstavlja logičke zahteve za expertima.



Conceptual request:



```text

Request {

&#x20;   request\_id

&#x20;   token\_id

&#x20;   layer

&#x20;   expert\_id

&#x20;   source

&#x20;   priority

}

```



`source` može biti:



```text

ACTUAL

PREDICTED

```



Phase 2a koristi prvenstveno:



```text

ACTUAL

```



i istoriju prethodnih tokena za residency odluke.



\---



\# 12. Request Deduplication



Više zahteva za isti expert ne sme proizvoditi više fizičkih promotion operacija.



Na primer:



```text

E5

E5

E5

E2

E5

```



treba da postane:



```text

E5 → one pending promotion

E2 → one pending promotion

```



Logical requests se vezuju za jednu fizičku operaciju.



```text

&#x20;            E5 request

&#x20;                 │

&#x20;                 ▼

&#x20;         pending promotion

&#x20;            /    |    \\

&#x20;           R1   R2    R3

```



\---



\# 13. Promotion Admission Policy



\*\*Svaki miss ne sme automatski pokrenuti promotion.\*\*



U suprotnom sistem može ući u PCIe thrashing:



```text

MISS

&#x20;→ copy

&#x20;→ expert unused

&#x20;→ copy another

&#x20;→ evict

&#x20;→ copy previous

&#x20;→ ...

```



Simulacija pokazuje da takav pristup može zauzeti približno sav raspoloživ PCIe transfer kapacitet.



Zato promotion zahteva admission policy.



Početna politika može biti:



```text

first miss

&#x20;   │

&#x20;   ▼

CPU fallback

&#x20;   │

&#x20;   ▼

record miss

```



Drugi miss istog eksperta u kratkom prozoru:



```text

second miss

&#x20;   │

&#x20;   ▼

promotion candidate

```



Alternativno:



```text

frequency(E) >= threshold

&#x20;   │

&#x20;   ▼

promotion candidate

```



Ovo pravilo treba biti konfigurabilno tokom eksperimentisanja.



\---



\# 14. CPU Fallback



CPU fallback nije samo error recovery.



On je sastavni deo scheduler dizajna.



```text

Expert miss

&#x20;   │

&#x20;   ▼

Is expert ready in VRAM?

&#x20;   │

&#x20;  NO

&#x20;   │

&#x20;   ├──────────────► CPU execution

&#x20;   │

&#x20;   └──────────────► admission accounting

```



Ako expert kasnije postane kandidat za promotion:



```text

CPU execution

&#x20;     │

&#x20;     ▼

promotion decision

&#x20;     │

&#x20;     ▼

background transfer

```



Na ovaj način CPU execution omogućava potpuno non-blocking ponašanje.



\---



\# 15. Promotion Queue



Nakon admission odluke expert ulazi u promotion queue.



```text

CPU resident

&#x20;    │

&#x20;    ▼

promotion admitted

&#x20;    │

&#x20;    ▼

Promotion Queue

&#x20;    │

&#x20;    ▼

VRAM slot allocation

&#x20;    │

&#x20;    ▼

background transfer

&#x20;    │

&#x20;    ▼

CUDA completion event

&#x20;    │

&#x20;    ▼

next token boundary

&#x20;    │

&#x20;    ▼

commit

```



\---



\# 16. Eviction



Ako nema slobodnog slot-a:



```text

Promotion request

&#x20;     │

&#x20;     ▼

free slot?

&#x20;  /      \\

&#x20;YES       NO

&#x20; │         │

&#x20; │         ▼

&#x20; │     select victim

&#x20; │         │

&#x20; │         ▼

&#x20; │       evict

&#x20; │         │

&#x20; └─────────┘

&#x20;      │

&#x20;      ▼

promotion

```



Početna politika:



```text

LRU

```



Kasnije je moguće uključiti:



```text

LRU

\+

frequency

\+

next-use information

\+

prediction confidence

\+

transfer cost

```



Ali eviction policy ne sme ugroziti correctness.



\---



\# 17. Token Boundary Commit



Kompletan lifecycle jedne promocije:



```text

Token t

──────────────────────────────────────



E5 = CPU\_RESIDENT



miss/accounting

&#x20;      │

&#x20;      ▼

admission accepted

&#x20;      │

&#x20;      ▼

promotion queued

&#x20;      │

&#x20;      ▼

CPU → VRAM transfer

&#x20;      │

&#x20;      ▼

CUDA event

&#x20;      │

&#x20;      ▼

transfer complete



──────────────────────────────────────

TOKEN BOUNDARY

──────────────────────────────────────



commit:

&#x20;   E5.location = VRAM

&#x20;   E5.slot      = X

&#x20;   E5.state     = VRAM\_RESIDENT



──────────────────────────────────────

Token t+1

──────────────────────────────────────



E5 may now be used as VRAM resident.

```



Ako transfer nije završen:



```text

TOKEN BOUNDARY

&#x20;   │

&#x20;   └── no commit

```



Expert ostaje CPU-resident.



\---



\# 18. CUDA Streams



Compute i transfer treba konceptualno razdvojiti:



```text

CUDA Compute Stream

────────────────────────────

GPU expert computation

GPU graph execution

merge

```



i:



```text

CUDA Transfer Stream

────────────────────────────

CPU → VRAM promotion

VRAM → CPU eviction

```



Transfer stream može raditi u pozadini dok compute stream obrađuje trenutni token, uz odgovarajuće CUDA event dependency-je.



\---



\# 19. Important llama.cpp / ggml Constraint



ggml graph se tretira kao execution structure sa stabilnim tensor/backing-memory odnosima tokom svog izvršavanja.



Dynamic residency zato ne treba konceptualno implementirati kao proizvoljno menjanje tensor backing storage-a usred aktivnog grafa.



Umesto toga:



```text

GGML GRAPH

&#x20;   │

&#x20;   ├── references stable execution resources

&#x20;   │

&#x20;   ▼

Residency / slot indirection

&#x20;   │

&#x20;   ▼

background transfer

```



Promena residency mapping-a mora biti sinhronizovana sa granicom na kojoj više ne postoji mogućnost da prethodni graph execution koristi stari slot.



Ovo je jedna od glavnih tačaka koju design review mora potvrditi na konkretnom llama.cpp/ggml kodu.



\---



\# 20. Phase 2a Execution Model



Kompletan tok:



```text

&#x20;                   TOKEN N

&#x20;                      │

&#x20;                      ▼

&#x20;                 MoE Router

&#x20;                      │

&#x20;                      ▼

&#x20;               Actual experts

&#x20;                      │

&#x20;                      ▼

&#x20;             Residency lookup

&#x20;                      │

&#x20;           ┌──────────┴──────────┐

&#x20;           ▼                     ▼

&#x20;       VRAM HIT              CPU MISS

&#x20;           │                     │

&#x20;           ▼                     ▼

&#x20;       GPU compute           CPU compute

&#x20;                                 │

&#x20;                                 ▼

&#x20;                        miss accounting

&#x20;                                 │

&#x20;                        ┌────────┴────────┐

&#x20;                        ▼                 ▼

&#x20;                   no admission       admitted

&#x20;                        │                 │

&#x20;                        │                 ▼

&#x20;                        │           promotion queue

&#x20;                        │                 │

&#x20;                        │                 ▼

&#x20;                        │          background copy

&#x20;                        │                 │

&#x20;                        │                 ▼

&#x20;                        │            CUDA event

&#x20;                        │                 │

&#x20;                        └────────┬────────┘

&#x20;                                 ▼

&#x20;                          TOKEN BOUNDARY

&#x20;                                 │

&#x20;                                 ▼

&#x20;                         commit completed

&#x20;                            residency

&#x20;                                 │

&#x20;                                 ▼

&#x20;                              TOKEN N+1

```



\---



\# 21. Phase 2b Predictor Model



Tek nakon validacije Phase 2a:



```text

Hidden State N

&#x20;     │

&#x20;     ▼

&#x20;  Predictor

&#x20;     │

&#x20;     ▼

Predicted experts N+1

&#x20;     │

&#x20;     ▼

Prefetch Queue

&#x20;     │

&#x20;     ▼

Residency Scheduler

```



Prediction confidence treba biti deo odluke.



Na primer:



```text

prediction confidence

&#x20;       │

&#x20;  ┌────┼────┐

&#x20;  ▼    ▼    ▼

&#x20;high  medium low

&#x20;  │    │     │

&#x20;  ▼    ▼     ▼

prefetch  maybe  ignore

```



Prediction nikada ne sme biti correctness dependency.



\---



\# 22. Performance Model



Phase 2 mora biti ocenjen u odnosu na Phase 1 baseline.



Definicija:



```text

Baseline = Phase 1 static residency

Candidate = Phase 2 dynamic residency

```



Minimum test matrix:



```text

Workload             Phase 1      Phase 2a

\------------------------------------------------

Serbian               T1           T2

Code                  T3           T4

Mixed Serbian+Code    T5           T6

```



Osnovni kriterijum:



```text

Mixed:

&#x20;   Phase 2a < Phase 1

```



odnosno bolje vreme/throughput.



Za pojedinačni domen:



```text

Phase 2a <= Phase 1

```



Dynamic residency ne sme degradirati workload samo zato što je uvedena dinamička politika.



Ako je određeni workload neutralan:



```text

Phase 2a ≈ Phase 1

```



to može biti prihvatljivo.



Ako je konzistentno sporiji:



```text

Phase 2a > Phase 1

```



design/policy se mora ponovo razmotriti pre Phase 2b.



\---



\# 23. Measurements



Za svaku konfiguraciju meriti:



```text

tokens/sec

token latency

prompt latency

GPU compute time

CPU fallback time

promotion count

promotion bytes

eviction count

eviction bytes

promotion hit rate

promotion usefulness

LRU hit rate

queue depth

queue wait time

PCIe transfer utilization

VRAM occupancy

```



Posebno:



```text

promotion usefulness =

promoted experts subsequently reused

/

all promoted experts

```



Ovo je važnije od samog broja promotion operacija.



\---



\# 24. Correctness Criteria



Phase 2 mora zadržati correctness karakteristike Phase 1.



Testirati najmanje:



```text

all-cold

all-hot

static mixed

dynamic residency

dynamic + CPU fallback

promotion miss

promotion hit

eviction

slot reuse

queue deduplication

multiple consecutive tokens

```



Rezultat mora ostati u okviru odgovarajućeg reference kriterijuma:



```text

CPU/RPC:

&#x20;   exact / bit-level



CUDA:

&#x20;   defined numerical tolerance / KLD

```



Posebno treba proveriti determinističnost oko token-boundary residency commit-a.



\---



\# 25. Safety Invariants



Sistem mora održavati sledeće invariantnosti:



\### Invariant 1



```text

GPU never executes from a slot

that is being overwritten.

```



\### Invariant 2



```text

Residency table never reports

VRAM\_RESIDENT before transfer completion.

```



\### Invariant 3



```text

Residency changes become visible

only at an allowed synchronization boundary.

```



\### Invariant 4



```text

A prediction failure cannot cause

correctness failure.

```



\### Invariant 5



```text

A promotion failure cannot prevent

CPU execution.

```



\### Invariant 6



```text

One expert cannot have multiple

conflicting physical residency operations.

```



\### Invariant 7



```text

Eviction cannot destroy an expert

while that expert is still in use.

```



\---



\# 26. Phase 2a Success Criteria



Phase 2a je uspešan samo ako zadovolji \*\*oba\*\* nivoa.



\## Functional



```text

✓ dynamic residency works

✓ slot reuse is safe

✓ token-boundary commit is correct

✓ CPU fallback works

✓ promotion works

✓ eviction works

✓ correctness preserved

```



\## Performance



```text

✓ mixed workload improves over Phase 1

✓ no tested domain is consistently degraded

✓ PCIe thrashing is controlled

✓ promotion usefulness is measurable

✓ queue overhead is measurable

```



Samo funkcionalno ispravan dynamic residency sistem nije dovoljan.



\---



\# 27. Phase 2b Entry Gate



Phase 2b predictor se ne implementira automatski.



Prelazak na Phase 2b zahteva dokaz da:



```text

Phase 2a

&#x20;   │

&#x20;   ▼

temporal locality

&#x20;   │

&#x20;   ▼

meaningful residency benefit

&#x20;   │

&#x20;   ▼

measured performance headroom

```



Tek tada:



```text

Phase 2b

&#x20;   │

&#x20;   ▼

prediction

&#x20;   │

&#x20;   ▼

intra-token prefetch

```



Ako Phase 2a već daje dovoljan rezultat, Phase 2b može biti nepotreban.



\---



\# 28. Design Review Deliverable



Pre bilo kakvog koda, design review treba da odgovori na sledeća pitanja:



1\. Kako llama.cpp trenutno predstavlja MoE expert tensor/backing storage?

2\. Gde je bezbedno uvesti residency indirection?

3\. Kako se fixed VRAM slots mogu povezati sa postojećim ggml tensor execution modelom?

4\. Koja je tačna granica token/layer execution-a na kojoj se residency table može menjati?

5\. Kako CUDA events garantuju bezbedan slot reuse?

6\. Kako se transfer stream može odvojiti od compute stream-a?

7\. Kako se CPU fallback povezuje sa postojećim Phase 1 putem?

8\. Gde se može implementirati request/promotion queue bez menjanja correctness puta?

9\. Koji je najbezbedniji minimalni LRU/admission model?

10\. Koje su konkretne prepreke za Phase 2a?

11\. Da li Phase 2b zahteva promene u graph scheduling-u?

12\. Koje delove llama.cpp/ggml-a bi implementacija morala da modifikuje?



\---



\# 29. Review Rule



Agentu se ovaj dokument prosleđuje kao:



\*\*Design Review — No Code\*\*



Agent treba da:



```text

READ

&#x20; ↓

ANALYZE llama.cpp / ggml architecture

&#x20; ↓

CHECK feasibility

&#x20; ↓

IDENTIFY conflicts

&#x20; ↓

IDENTIFY synchronization points

&#x20; ↓

PROPOSE implementation plan

```



Agent \*\*ne treba da menja kod\*\* u ovoj fazi.



Implementacija počinje tek nakon:



```text

Phase 1 measurements

&#x20;       +

Design Review

&#x20;       +

feasibility confirmation

```



\---



\# 30. Core Principle



Cela Phase 2 može se svesti na sledeći princip:



```text

&#x20;                   ┌──────────────┐

&#x20;                   │    Router    │

&#x20;                   └──────┬───────┘

&#x20;                          │

&#x20;                   actual expert

&#x20;                          │

&#x20;                          ▼

&#x20;                 ┌─────────────────┐

&#x20;                 │ Residency Table │

&#x20;                 └────────┬────────┘

&#x20;                          │

&#x20;             ┌────────────┼────────────┐

&#x20;             ▼            ▼            ▼

&#x20;          VRAM HIT      CPU MISS     PENDING

&#x20;             │            │            │

&#x20;             ▼            ▼            ▼

&#x20;            GPU          CPU       background

&#x20;          compute       compute     transfer

&#x20;                                        │

&#x20;                                        ▼

&#x20;                                   CUDA event

&#x20;                                        │

&#x20;                                        ▼

&#x20;                                 TOKEN BOUNDARY

&#x20;                                        │

&#x20;                                        ▼

&#x20;                                 residency commit

```



\*\*The GPU never waits for residency.\*\*



Dynamic residency pokušava da pripremi buduće eksperte u pozadini, ali trenutni compute uvek ima funkcionalan put kroz već postojeći CPU fallback.



Phase 2a prvo koristi \*\*temporal locality između tokena\*\*. Phase 2b tek kasnije uvodi \*\*prediction i intra-token prefetch\*\*, ako merenja pokažu da dodatna kompleksnost donosi merljiv benefit.



Konačna odluka o implementaciji treba da bude zasnovana na rezultatima Phase 1 benchmarka i design review-u konkretne llama.cpp/ggml arhitekture.



