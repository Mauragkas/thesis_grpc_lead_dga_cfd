---
title: "Έκθεση Προόδου Διπλωματικής Εργασίας"
subtitle: "Κατανεμημένος Γενετικός Αλγόριθμος για τη Βελτιστοποίηση της Αεροδυναμικής Αεροσκάφους με χρήση LEAD DHT και Ιεραρχικού Μοντέλου Προσεγγιστικής Αξιολόγησης (Multi-Tier Surrogate)"
author: |
  \vspace{0.3cm}
  \parbox{\linewidth}{\centering
  \Large \textbf{Χριστοδουλόπουλος Ευστάθιος Παναγιώτης} \\
  \vspace{0.2cm}
  \normalsize ΑΜ: 1093513 \quad $\cdot$ \quad \texttt{up1093513@ac.upatras.gr} \quad $\cdot$ \quad Έτος: 5ο \\
  \vspace{0.1cm}
  \normalsize Τμήμα Μηχανικών Η/Υ \& Πληροφορικής, Πανεπιστήμιο Πατρών
  }
date: "Αύγουστος 2026"
geometry: "left=0.6in,right=0.6in,top=0.7in,bottom=0.7in"
fontsize: 10pt
header-includes: |
  \usepackage{booktabs}
  \usepackage{xcolor}
  \usepackage{amsmath}
  \usepackage{amssymb}
  \usepackage{graphicx}
  \usepackage{float}
  \usepackage{caption}
  \usepackage{hyperref}
  \hypersetup{colorlinks=true, linkcolor=blue, urlcolor=blue}
  \captionsetup{font=small, labelfont=bf}
---

# 1. Εισαγωγή & Σκοπός της Έκθεσης

Η παρούσα έκθεση αποσκοπεί στην καταγραφή της συνολικής προόδου που έχει σημειωθεί στην υλοποίηση της διπλωματικής εργασίας. Παρουσιάζει:
1. Τη **βασική ιδέα** και το πρόβλημα που επιλύουμε.
2. Την **εξέλιξη της αρχιτεκτονικής** σε σχέση με την αρχική πρόταση (μετάβαση από LSH-DHT σε **LEAD DHT**, επιλογή **Rust** & **AeroSandbox**, ιεραρχική αξιολόγηση **3-Tier**).
3. Την **αναλυτική κατάσταση των υποσυστημάτων** (Orchestrator, LEAD Node, Surrogate Microservice, Worker Simulator, Monitor Dashboard).
4. Τα **τεχνικά workarounds** και τις σχεδιαστικές αποφάσεις που εφαρμόστηκαν.
5. Τον **βαθμό ολοκλήρωσης** και τα εναπομείναντα παραδοτέα.
6. **Ερωτήματα και σημεία προς συζήτηση / καθοδήγηση** από τον επιβλέποντα καθηγητή.

---

## 2. Σύνοψη της Βασικής Ιδέας & του Προβλήματος

Ο αεροδυναμικός σχεδιασμός ενός αεροσκάφους (και ειδικότερα ενός 3D-εκτυπώσιμου μη επανδρωμένου αεροσκάφους — UAV) αποτελεί ένα σύνθετο πρόβλημα βελτιστοποίησης πολλών παραμέτρων. Η παραδοσιακή αξιολόγηση μέσω Υπολογιστικής Ρευστοδυναμικής (CFD) είναι υπολογιστικά εξαιρετικά βαριά: μία πλήρης προσομοίωση μπορεί να διαρκέσει από αρκετά λεπτά έως ώρες. Αυτό καθιστά την εκτέλεση ενός κλασικού Γενετικού Αλγορίθμου (GA) —ο οποίος απαιτεί χιλιάδες αξιολογήσεις— ασύμφορη ή χρονικά ανέφικτη.

### Η Προτεινόμενη Λύση
Για να ξεπεραστεί αυτό το εμπόδιο, αναπτύσσουμε ένα πλήρως κατανεμημένο, συνεργατικό σύστημα βελτιστοποίησης που συνδυάζει:
- **Μοντέλο Νήσων (Island Model DGA):** Ο πληθυσμός των υποψηφίων αεροσκαφών διαμοιράζεται σε ανεξάρτητους εξελικτικούς κόμβους (νήσους / orchestrators), οι οποίοι επικοινωνούν σε δακτύλιο (Chord-like ring) και ανταλλάσσουν τα καλύτερα άτομα (μετανάστευση / migration) για αποφυγή τοπικών βέλτιστων.
- **Κατανεμημένο Ευρετήριο Μάθησης (LEAD DHT):** Αντί να υπολογίζουμε από το μηδέν κάθε νέο σχέδιο, διατηρούμε μια κατανεμημένη χωρική μνήμη. Όταν ο αλγόριθμος παράγει ένα σχέδιο που μοιάζει γεωμετρικά με κάποιο ήδη αξιολογημένο, ανακτούμε άμεσα το αποτέλεσμα ή εκτιμούμε την καταλληλότητά του.
- **Ιεραρχικό Σύστημα Αξιολόγησης 3 Επιπέδων (Multi-Tier $\epsilon$-Bypass Pipeline):**
  1. **Tier 1 (Άμεση Ανάκτηση / Cache Hit):** Αν το σχέδιο είναι σχεδόν ταυτόσημο ($d_{\min} < \epsilon_{\text{exact}}$), επιστρέφεται άμεσα η γνωστή τιμή (μηδενικό υπολογιστικό κόστος).
  2. **Tier 2 (Προσεγγιστικό Μοντέλο / Surrogate Model):** Αν το σχέδιο βρίσκεται σε γνωστή γειτονιά ($\epsilon_{\text{exact}} \le d_{\min} \le R$), η καταλληλότητα προβλέπεται ταχύτατα από ένα ενεργό νευρωνικό δίκτυο (MLP) ή μοντέλο Gaussian Process.
  3. **Tier 3 (Πλήρης Προσομοίωση):** Μόνο αν το σχέδιο είναι εντελώς νέο / ανεξερεύνητο ($d_{\min} > R$), αποστέλλεται για πραγματική προσομοίωση σε κατανεμημένους εξομοιωτές (workers), και το νέο αποτέλεσμα τροφοδοτεί πίσω το DHT και το surrogate μοντέλο.

```
                         Υποψήφιο Αεροσκάφος x*
                                    │
                                    ▼
                     [Υπολογισμός Ελάχιστης Απόστασης d_min]
                                    │
                  ┌─────────────────┼─────────────────┐
                  ▼                 ▼                 ▼
            d_min < ε_exact   ε_exact ≤ d_min ≤ R   d_min > R
            [ Tier 1: Hit ]  [ Tier 2: Surrogate ] [ Tier 3: True Sim ]
            Επιστροφή y_NN   Πρόβλεψη μέσω MLP     Εκτέλεση σε Worker
            (0 υπολογισμός)  Microservice (gRPC)   Αποθήκευση στο LEAD
                                                   & Online Retraining
```

---

## 3. Εξέλιξη της Αρχιτεκτονικής (Αρχική Πρόταση vs Πραγματικότητα)

Κατά τη διάρκεια της ανάπτυξης και των πειραματικών δοκιμών, πραγματοποιήθηκαν ορισμένες κρίσιμες σχεδιαστικές αναβαθμίσεις σε σχέση με το αρχικό concept paper:

| Τομέας | Αρχική Πρόταση (Concept PDF) | Τελική Υλοποίηση στο Σύστημα | Αιτιολόγηση / Όφελος |
|---|---|---|---|
| **Distributed Hash Table** | Κλασικό LSH-DHT (Locality Sensitive Hashing) | **LEAD DHT (Learned Indexing + Hilbert Space-Filling Curve)** | Το LSH έχει στατικά «κουτιά» (buckets) και δεν υποστηρίζει αποδοτικά range queries σε κατανεμημένο δακτύλιο. Το LEAD χαρτογραφεί τον 10D χώρο γεωμετρίας σε 1D μέσω καμπύλης Hilbert και χρησιμοποιεί RMI μοντέλο με PID controller για δυναμική προσαρμογή στην κατανομή. |
| **Γλώσσα Υλοποίησης Κόμβων** | Go + gRPC | **Rust (100%) + Tonic gRPC + Tokio** | Η Rust εξασφαλίζει μηδενικό runtime overhead, απουσία παύσεων Garbage Collector (κρίσιμο για σταθερά χαμηλά latencies στο DHT και στον Orchestrator), και απόλυτη ασφάλεια μνήμης. |
| **Αεροδυναμική Προσομοίωση** | OpenFOAM (snappyHexMesh / RANS) | **AeroSandbox (3D Vortex Lattice Method - VLM) σε Python Workers** | Το OpenFOAM απαιτεί 5–20 λεπτά ανά γεωμετρία, καθιστώντας τα evolutionary runs (χιλιάδες αξιολογήσεις) απαγορευτικά για δοκιμές. Το AeroSandbox παρέχει ρεαλιστική αεροδυναμική (L/D, cruise trim, static pitch margin) σε λιγότερο. |
| **Surrogate Modeling** | Απλό ML / Python scikit-learn | **Αυτόνομο Microservice `surrogate_node` (Rust + C++ / CUDA / OpenMP)** | Υλοποιήθηκαν 4 αρχιτεκτονικές (MLP, Gaussian Process, Random Forest, k-NN) με hardware acceleration, sliding-window buffer και online retraining στο παρασκήνιο με zero-downtime atomic swap. |
| **Load Balancing** | LSH-DHT ως balancer | **Envoy Proxy (gRPC Round-Robin + Dynamic DNS)** | Ο Envoy διαχειρίζεται αξιόπιστα τη δρομολόγηση των αιτημάτων αξιολόγησης προς τους Python workers με αυτόματη ανακάλυψη κλιμακούμενων containers (`--scale worker=N`). |
| **Παρακολούθηση (Monitoring)** | Βασικό Node.js UI | **Event-Driven Log Streaming (Fluent-Bit $\to$ Kafka KRaft $\to$ Node.js / WS Dashboard)** | Κεντρική συλλογή structured JSON logs από όλους τους κόμβους, φιλτράρισμα ανά υποσύστημα και προβολή σε πραγματικό χρόνο μέσω WebSockets. |

---

# 4. Αναλυτική Κατάσταση Υποσυστημάτων

![Συνολική Αρχιτεκτονική Κατανεμημένου Συστήματος Multi-Island GA](compose/architecture.png){ width=85% }

### 4.1. Orchestrator (`orchestrator/` --- Rust)
- **Island Model GA:** Εκτελεί τον εξελικτικό κύκλο με επιλογή $(\mu + \lambda)$ elitism, γκαουσιανή μετάλλαξη και συνεχή έλεγχο ορίων γονιδίων (clipping).
- **Chord Ring & Migration:** Οι orchestrators σχηματίζουν δικό τους δακτύλιο peer-to-peer (μέσω SHA-256 διευθύνσεων) και ανταλλάσσουν τους κορυφαίους μετανάστες ανά προκαθορισμένο αριθμό γενεών.
- **Multi-Tier Evaluator:** Υλοποιεί πλήρως τη λογική $d_{\min} < \epsilon_{\text{exact}}$ (Tier 1), $\epsilon_{\text{exact}} \le d_{\min} \le R$ (Tier 2), $d_{\min} > R$ (Tier 3), με real-time καταγραφή μετρικών bypass ratio.
- **Gene Store & Neighbor Store:** Διαθέτει τοπική μνήμη με αυτόματη εκκαθάριση παλαιών γενεών (`GenerationEvictor`) και σύνδεση με το κατανεμημένο LEAD DHT μέσω πολυ-σημειακής καμπύλης Hilbert (`multi-probe Hilbert`).

### 4.2. LEAD DHT Node (`lead/` --- Rust)
- **Chord Ring με Virtual Nodes (vnodes):** Κάθε φυσικός κόμβος διαχειρίζεται 20 εικονικούς κόμβους (συνολικά 60 vnodes στον τριμελή δακτύλιο) για ομοιόμορφη κατανομή φορτίου.
- **Learned Indexing (RMI):** Μοντέλο δύο επιπέδων (Stage-0 bins $\to$ Stage-1 leaf models) που μετατρέπει τις 10D συντεταγμένες σε 64-bit θέσεις δακτυλίου διατηρώντας τη διάταξη.
- **Online PID Controller:** Αυτόματος ρυθμιστής 2-bit που τροποποιεί δυναμικά τα όρια (scale & offset) των φύλλων του μοντέλου ώστε να διατηρείται ακρίβεια δρομολόγησης $\ge 95\%$.
- **Concept Drift & Federated Averaging (FedAvg):** Όταν τα νέα κλειδιά προκαλούν μετατόπιση κατανομής ($>40\%$), εκλέγεται προσωρινός συντονιστής μέσω heartbeat consensus, συνενώνονται τα τοπικά μοντέλα μέσω FedAvg, και εκτελείται αναδιανομή των κλειδιών (`key migration`).
- **Ordered Range Queries:** Υποστήριξη αναζητήσεων εύρους μεταξύ διαδοχικών κόμβων με προστασία από κυκλικές αναδρομές.

### 4.3. Surrogate Microservice (`surrogate_node/` --- Rust + C++/CUDA/OpenMP)
- **Υποστηριζόμενα Μοντέλα:**
  1. *Multi-Layer Perceptron (MLP)* με SiLU/ReLU ενεργοποιήσεις και Adam optimizer (προεπιλογή για online inference).
  2. *Gaussian Process Regression (GP)* με Matérn 5/2 & RBF kernels και Cholesky factorization.
  3. *Random Forest Regression (RF)* με πολυνηματική εκτίμηση αβεβαιότητας.
  4. *Exact k-Nearest Neighbors (k-NN)*.
- **Sliding Window Buffer:** FIFO buffer με όριο δειγμάτων (π.χ. 1.000) για αποθήκευση μόνο των πιο πρόσφατων και έγκυρων γεωμετριών.
- **Online Retraining:** Επαναληπτική εκπαίδευση στο παρασκήνιο κάθε $N$ νέα δείγματα και άμεση, αδιάλειπτη εναλλαγή ενεργού μοντέλου (`atomic model swapping`).

### 4.4. Worker Simulation Service (`worker/` --- Python + AeroSandbox)
- **Παραμετρικό Μοντέλο Αεροσκάφους (10 Γονίδια):**
  - Γεωμετρία πτέρυγας: εκπέτασμα ($b$), χορδή ρίζας ($c_{\text{root}}$), χορδή ακροπτερυγίου ($c_{\text{tip}}$), γωνία βέλους ($\Lambda$), δίεδρο ($\Gamma$), συστροφή ($\theta_{\text{twist}}$), θέση $X$.
  - Αεροτομή πτέρυγας: 4-ψήφια παραμετρική NACA (`nacaMPTT` --- μέγιστη καμπυλότητα, θέση, πάχος).
  - Ουραίο πτέρωμα: οριζόντιο \& κατακόρυφο σταθερό με συμμετρική αεροτομή NACA 0010.
  - Άτρακτος: υπολογισμός διαθέσιμου εσωτερικού όγκου για payload ($V_{\text{fuse}} \ge 20.000\text{ mm}^3$).
- **Επίλυση VLM & Cruise Trim:** Εκτέλεση σάρωσης 13 σημείων $\alpha \in [-2^\circ, 10^\circ]$ μέσω AeroSandbox, υπολογισμός σημείου ισορροπίας πτήσης ($C_L = C_{L,\text{req}}$), εκτίμηση λόγου $L/D$, και ποινή στατικής διαμήκους ευστάθειας ($C_{m_\alpha} = \frac{dC_m}{d\alpha}$).

### 4.5. Monitor Dashboard & Infrastructure (`monitor/`, `compose/`)
- **Docker Compose Topology:** Πλήρως απομονωμένο δίκτυο `simulated-lan` με δυνατότητα εκτέλεσης σε διαχωρισμένα compose files (infra, lead, worker, worker2).
- **Log Aggregation:** Παραγωγή δομημένων JSON logs από όλα τα υποσυστήματα $\to$ Fluent-Bit $\to$ Kafka topics $\to$ Node.js/WebSocket web dashboard.
- **Web UI:** Περιβάλλον παρακολούθησης με φιλτράρισμα πηγών (Lead, Orchestrator, Worker, Surrogate), επίπεδα σφαλμάτων, search bar και pagination.

---

# 5. Τεχνικά Workarounds & Σχεδιαστικές Αποφάσεις

1. **AeroSandbox αντί για OpenFOAM:** Η αυτόματη δημιουργία πλέγματος (meshing) και η επίλυση Navier-Stokes στο OpenFOAM αποδείχθηκε εξαιρετικά ευάλωτη σε ακραίες γεωμετρίες του GA (divergence / mesh errors) και υπερβολικά αργή. Το AeroSandbox είναι ντετερμινιστικό, ταχύτατο (~100ms) και επιστρέφει έγκυρες τιμές για $L/D$ και $C_{m_\alpha}$.
2. **Αποφυγή Ατέρμονων Βρόχων στο LEAD DHT:** Σε αρχικό στάδιο (cold start), όταν το DHT δεν είχε ακόμη αποθηκευμένα κλειδιά, τα range queries προωθούνταν κυκλικά μεταξύ των διαδοχικών vnodes επ' άπειρον. Ενσωματώθηκε μηχανισμός ανίχνευσης κύκλου (`origin_vid` \& tracking επισκεφθέντων κόμβων), τερματίζοντας άμεσα την αναζήτηση.
3. **Ασφαλής Διαχείριση Cold Start στο Surrogate:** Στις πρώτες γενιές, το νευρωνικό δίκτυο δεν διαθέτει επαρκή δείγματα ($\ge 30$) για να θεωρηθεί αξιόπιστο. Υλοποιήθηκε λογική graceful fallback: αν το surrogate αναφέρει ότι δεν είναι έτοιμο (`not ready`), όλα τα άτομα του Tier 2 δρομολογούνται αυτόματα στο Tier 3 (πραγματικοί workers).
4. **Multi-Probe Hilbert Encoding:** Οι καμπύλες πλήρωσης χώρου εμφανίζουν ασυνέχειες στα όρια των υποδιαιρέσεων, με αποτέλεσμα δύο κοντινά σημεία στον 10D χώρο να καταλήγουν σε απομακρυσμένες 1D τιμές. Εφαρμόστηκε τεχνική Multi-Probe με περιστραμμένες καμπύλες Hilbert, αναζητώντας γείτονες σε πολλαπλές παραλλαγές.

---

# 6. Πού Βρισκόμαστε Σήμερα (Βαθμός Ολοκλήρωσης)

### Ολοκληρώθηκαν Πλήρως 100\%
- **Πυρήνας Γενετικού Αλγορίθμου:** Island Model, $\mu+\lambda$ selection, Gaussian mutation, migration ring.
- **Κατανεμημένο DHT LEAD:** Chord ring, 60 vnodes, RMI Learned Index, 2-bit PID tuner, FedAvg, range queries.
- **Ιεραρχικός Αξιολογητής 3 Επιπέδων:** Tier 1 $\epsilon$-bypass, Tier 2 Surrogate, Tier 3 Simulation \& feedback loop.
- **Υπηρεσία Surrogate:** `surrogate_node` με MLP, Gaussian Process, Random Forest, k-NN, sliding window \& online training.
- **Αεροδυναμικός Worker:** Python με AeroSandbox (10D γεωμετρία, VLM cruise trim, stability penalty, volume check).
- **Υποδομή Microservices:** Docker Compose, Envoy gRPC Load Balancer, Kafka KRaft, Fluent-Bit.
- **Σουίτα Αυτοματοποιημένων Δοκιμών:** Rust `cargo test`, Python `pytest`, Criterion benchmarks, integration tests `test_sim.py`, `test_dht.py`, `test_gp.py`.

### Σε Εξέλιξη / Εκκρεμότητες
- **Συνεννόηση με Εργαστήριο MEAD για Αεροδυναμική:** Οριστικοποίηση των αεροδυναμικών παραμέτρων, συνθηκών πτήσης cruise και πιθανής επικύρωσης (validation) σε συνεργασία με το εργαστήριο του Τμήματος Μηχανολόγων \& Αεροναυπηγών Μηχανικών (MEAD).
- **Τελικά Πειράματα Κλιμάκωσης & Μετρήσεις Bypass:** Εκτέλεση συστηματικών sweeps για:
  - Επίδραση των υπερπαραμέτρων $\epsilon_{\text{exact}}$ και $R$ στο τελικό Bypass Ratio (\%) και στην ταχύτητα σύγκλισης.
  - Κλιμάκωση αριθμού workers ($N = 1, 2, 4, 8, 12, 16$) και μέτρηση του speedup σε σχέση με απλό GA χωρίς LEAD/Surrogate.
  - Σύγκριση ακρίβειας πρόβλεψης και χρόνου εκπαίδευσης μεταξύ MLP και Gaussian Process στο surrogate buffer.
- **Συγγραφή του Κειμένου της Διπλωματικής Εργασίας (Τόμος):** Καταγραφή του θεωρητικού υποβάθρου, της αρχιτεκτονικής ανάλυσης, των use cases, sequence diagrams και των πειραματικών γραφημάτων.
- **UI / Monitoring Dashboard (Χαμηλή Προτεραιότητα):** Το backend συλλέγει και δρομολογεί πλήρως τα logs μέσω Kafka/WebSockets. Η περαιτέρω αισθητική βελτίωση του web dashboard και η προσθήκη Three.js visualizer θα γίνει σε μεταγενέστερο στάδιο (polish phase).

---

# 7. Ερωτήματα & Σημεία για Καθοδήγηση (Seeking Guidance)

Παρακαλώ για τις παρατηρήσεις και την καθοδήγησή σας στα παρακάτω κρίσιμα σημεία της κατανεμημένης αρχιτεκτονικής και των πειραμάτων:

1. **Πειραματικά Σενάρια Αξιολόγησης:**  
   *Ερώτημα:* Ποιες συγκρίσεις θεωρείτε πιο ουσιώδεις για το κεφάλαιο των αποτελεσμάτων; Προτείνουμε τα εξής σενάρια:
   - *Σενάριο Α:* Κλασικός GA χωρίς Caching (Baseline).
   - *Σενάριο Β:* GA + LEAD DHT (Tier 1 Exact Match Bypass).
   - *Σενάριο Γ:* GA + LEAD DHT + MLP Surrogate (Πλήρες 3-Tier Pipeline).
   - *Σενάριο Δ:* Μελέτη ευαισθησίας παραμέτρων $\epsilon$ και $R$ (Bypass Ratio vs Ακρίβεια Βέλτιστης Λύσης).
   - *Σενάριο Ε:* Μελέτη κλιμάκωσης (Scalability Benchmark με 1 έως 16 workers).

2. **Υπερπαράμετροι Γενετικού Αλγορίθμου & Μετανάστευσης:**  
   *Ερώτημα:* Υπάρχει κάποια συγκεκριμένη τοπολογία ή ρυθμός μετανάστευσης (π.χ. μεταφορά του 10% των αρίστων κάθε 5 γενιές σε δακτύλιο) που θα προτείνατε να υιοθετηθεί ως standard για τις τελικές συγκρίσεις;

3. **Συνεργασία με Εργαστήριο MEAD:**  
   *Σημείωση:* Για τις αεροδυναμικές λεπτομέρειες, τις επιτρεπτές τιμές γεωμετρίας και τις συνθήκες πτήσης του μοντέλου, θα πραγματοποιηθεί συνεννόηση με το εργαστήριο του MEAD, ώστε το μοντέλο αξιολόγησης να συμφωνεί με τις πρακτικές προδιαγραφές 3D εκτύπωσης και δοκιμών.

---
