# Curated Academic Literature & Reference Guide
**Thesis:** *Distributed Genetic Algorithm for Aircraft Aerodynamic Optimization using LEAD DHT and Multi-Tier Surrogate Evaluation Model*  
**Author:** Efstathios Panagiotis Christodoulopoulos  
**Department:** Computer Engineering & Informatics Department (CEID), University of Patras  

---

## Overview of the Curated Literature

This document aggregates the most relevant academic literature (rated **Moderate**, **High**, and **Core Foundation**) aligned with the thesis architecture:
1. **Pillar 1: Distributed Hash Tables, Learned Indexing & Spatial Curves** (LEAD DHT, Chord, RMI, Hilbert curves)
2. **Pillar 2: Surrogate-Assisted Evolutionary Algorithms (SAEA) & Multi-Tier Pipelines** (Gaussian Processes, Kriging, Localized Regression)
3. **Pillar 3: Distributed Genetic Algorithms & Island Models** (Ring migration topologies, rumor spreading, elitism)
4. **Pillar 4: Aerodynamics, 3D Wing Optimization & Geometric Parameterization** (AeroSandbox, VLM, AVL, OptiWing3D, CST, NACA)
5. **Pillar 5: Systems Infrastructure, Rust & Distributed Microservices** (Rust distributed systems, zero GC overhead)

---

## Pillar 1: Distributed Hash Tables, Learned Indexing & Spatial Hashing

### 1. A Distributed Learned Hash Table (LEAD DHT)
* **Authors:** Shengze Wang, Yi Liu, Xiaoxue Zhang, Liting Hu, Chen Qian
* **Publication:** arXiv:2508.14239 / *IEEE Transactions on Networking* (2025/2026)
* **Relevance:** **Core Foundation (Essential)**
* **Link:** [https://arxiv.org/abs/2508.14239](https://arxiv.org/abs/2508.14239) | [PDF](https://arxiv.org/pdf/2508.14239v1.pdf)
* **Role in Thesis:** Primary reference for the `lead/` crate. Defines the order-preserving Learned Hash function using Recursive Model Indexing (RMI), the Federated Recursive Model (FRM) with FedAvg for handling distribution drift, and the Shadow Balancer using virtual nodes ($k=20$). Cite in **Chapter 2 (Background)** and **Chapter 3 (System Architecture)**.

### 2. The Case for Learned Index Structures
* **Authors:** Tim Kraska, Alex Beutel, Ed H. Chi, Jeffrey Dean, Neoklis Polyzotis
* **Publication:** *Proceedings of the 2018 ACM SIGMOD International Conference on Management of Data*, pp. 489–504 (2018)
* **Relevance:** **Core Foundation (Essential)**
* **Link:** [https://dl.acm.org/doi/10.1145/3183713.3196909](https://dl.acm.org/doi/10.1145/3183713.3196909) | [RMI GitHub Reference](https://github.com/learnedsystems/rmi)
* **Role in Thesis:** The seminal paper introducing Recursive Model Indexes (RMI). Replaces classical B-Trees and hash tables with functional CDF approximations. Directly cited in **Chapter 2 & Chapter 4** for the 2-stage RMI implementation in `src/rmi/`.

### 3. Chord: A Scalable Peer-to-Peer Lookup Service for Internet Applications
* **Authors:** Ion Stoica, Robert Morris, David Karger, M. Frans Kaashoek, Hari Balakrishnan
* **Publication:** *ACM SIGCOMM Computer Communication Review*, 31(4), 149–160 (2001) / *IEEE/ACM Transactions on Networking* (2003)
* **Relevance:** **Core Foundation (Essential)**
* **Link:** [https://pdos.csail.mit.edu/papers/chord:sigcomm01/chord_sigcomm.pdf](https://pdos.csail.mit.edu/papers/chord:sigcomm01/chord_sigcomm.pdf)
* **Role in Thesis:** Theoretical foundation of consistent hashing on a 64-bit ring, finger tables ($O(\log N)$ routing), and successor stabilization upon which LEAD DHT and the orchestrator migration ring are constructed. Cite in **Chapter 2**.

### 4. Multi-Probe LSH: Efficient Indexing for High-Dimensional Similarity Search
* **Authors:** Qin Lv, William Josephson, Zhe Wang, Moses Charikar, Kai Li
* **Publication:** *Proceedings of the 33rd International Conference on Very Large Data Bases (VLDB)*, pp. 950–961 (2007)
* **Relevance:** **High**
* **Link:** [https://www.vldb.org/conf/2007/papers/research/p950-lv.pdf](https://www.vldb.org/conf/2007/papers/research/p950-lv.pdf)
* **Role in Thesis:** Theoretical justification for multi-probing. Adapted in `hilbert/` to overcome boundary discontinuities where nearby points in 10D geometry space hash to disjoint segments on a 1D space-filling curve. Cite in **Chapter 3 & 4**.

### 5. Programming the Hilbert Curve
* **Authors:** John Skilling
* **Publication:** *AIP Conference Proceedings*, 707(1), 381–387 (2004)
* **Relevance:** **Core Foundation (Essential)**
* **Link:** [https://doi.org/10.1063/1.1751381](https://doi.org/10.1063/1.1751381)
* **Role in Thesis:** Provides the exact bit-manipulation algorithm for arbitrary dimensions and bit-depths used in `hilbert_rs` (`hilbert/src/lib.rs`). Cite in **Chapter 4 (Implementation)**.

### 6. Using Space-Filling Curves for Multi-Dimensional Indexing
* **Authors:** Jonathan K. Lawder, Peter J. H. King
* **Publication:** *British National Conference on Databases (BNCOD)*, Springer, pp. 20–35 (2000)
* **Relevance:** **High**
* **Link:** [https://doi.org/10.1007/3-540-45062-8_3](https://doi.org/10.1007/3-540-45062-8_3)
* **Role in Thesis:** Theoretical proof of locality preservation and spatial clustering when mapping multidimensional vector spaces to 1D keys. Cite in **Chapter 2 & 3**.

### 7. Learning Multi-Dimensional Indexes (Flood)
* **Authors:** Vikram Nathan, Jialin Ding, Mohammad Alizadeh, Tim Kraska
* **Publication:** *Proceedings of the 2020 ACM SIGMOD International Conference on Management of Data*, pp. 985–1000 (2020)
* **Relevance:** **Moderate / Contextual**
* **Link:** [https://dl.acm.org/doi/10.1145/3318464.3389714](https://dl.acm.org/doi/10.1145/3318464.3389714)
* **Role in Thesis:** Demonstrates the limitations of standard 1D learned indexes when extended to multi-dimensional data, providing academic justification for combining Hilbert space-filling curves with RMI. Cite in **Chapter 2 (Related Work)**.

### 8. Learned Metric Index in Rust
* **Author:** David Procházka (Advisor: V. Dohnal)
* **Publication:** *Master's Thesis, Faculty of Informatics, Masaryk University* (2024)
* **Relevance:** **High / Related Implementation**
* **Link:** [https://is.muni.cz/th/tmmii/thesis.pdf](https://is.muni.cz/th/tmmii/thesis.pdf)
* **Role in Thesis:** Directly related implementation of learned similarity and metric indexing in Rust. Validates language suitability and provides an empirical reference point for Rust-based learned indices. Cite in **Chapter 2 & 4**.

### 9. LIDER: An Efficient High-Dimensional Learned Index for Large-Scale Dense Retrieval
* **Authors:** Yifan Wang et al.
* **Publication:** *ResearchGate / Information Retrieval* (2022)
* **Relevance:** **Moderate**
* **Link:** [https://www.researchgate.net/publication/365706495_LIDER_an_efficient_high-dimensional_learned_index_for_large-scale_dense_passage_retrieval](https://www.researchgate.net/publication/365706495_LIDER_an_efficient_high-dimensional_learned_index_for_large-scale_dense_passage_retrieval)
* **Role in Thesis:** Useful for discussing high-dimensional feature partitioning and learned approximation vs exact search. Cite in **Chapter 2**.

### 10. Complementary Hashing for Approximate Nearest Neighbor Search
* **Authors:** Hao Xu, Jingdong Wang, Xian-Sheng Hua, Shipeng Li
* **Publication:** *IEEE International Conference on Computer Vision (ICCV)*, pp. 1631–1638 (2011)
* **Relevance:** **Moderate**
* **Link:** [https://www.microsoft.com/en-us/research/wp-content/uploads/2017/01/ICCV11-hashing.pdf](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/01/ICCV11-hashing.pdf)
* **Role in Thesis:** Discusses multiple complementary hash tables to reduce missed near-neighbors in continuous spaces. Cite in **Chapter 2 (Nearest Neighbor Search)**.

---

## Pillar 2: Surrogate-Assisted Evolutionary Optimization & Multi-Tier Pipelines

### 11. Surrogate-Assisted Evolutionary Computation: Recent Advances and Future Challenges
* **Author:** Yaochu Jin
* **Publication:** *Swarm and Evolutionary Computation*, 1(2), 61–70 (2011)
* **Relevance:** **Core Foundation (Essential)**
* **Link:** [https://doi.org/10.1016/j.swevo.2011.05.001](https://doi.org/10.1016/j.swevo.2011.05.001)
* **Role in Thesis:** The authoritative survey paper on Surrogate-Assisted Evolutionary Algorithms (SAEAs). Defines global vs. local surrogates, model management, and online model updating. Cite in **Chapter 2 & 3**.

### 12. A Framework for Evolutionary Optimization with Approximate Fitness Functions
* **Authors:** Yaochu Jin, Markus Olhofer, Bernhard Sendhoff
* **Publication:** *IEEE Transactions on Evolutionary Computation*, 6(5), 481–494 (2002)
* **Relevance:** **Core Foundation (Essential)**
* **Link:** [https://doi.org/10.1109/TEVC.2002.800884](https://doi.org/10.1109/TEVC.2002.800884)
* **Role in Thesis:** Provides the theoretical framework for individual-based evolution control: evaluating only a fraction of candidate solutions with expensive true solvers while bypassing or approximating the rest. Directly justifies your 3-Tier Multi-Tier Evaluator. Cite in **Chapter 3**.

### 13. Locally Weighted Regression Models for Surrogate-Assisted Design Optimization
* **Authors:** R. Le Riche et al.
* **Publication:** *Optimization Online* (2016)
* **Relevance:** **High**
* **Link:** [https://optimization-online.org/2016/11/5729/](https://optimization-online.org/2016/11/5729/)
* **Role in Thesis:** Mathematical justification for localized surrogate boundaries ($d_{\min} \le R$), demonstrating that local surrogates in design space outperform global approximators when data is non-uniform. Cite in **Chapter 3**.

### 14. Surrogates: Gaussian Process Modeling, Design, and Optimization for the Applied Sciences
* **Author:** Robert B. Gramacy
* **Publication:** *Chapman & Hall/CRC Press*, 584 pages (2020)
* **Relevance:** **Core Foundation (Essential)**
* **Link:** [https://bookdown.org/rbg/surrogates/chap5.html](https://bookdown.org/rbg/surrogates/chap5.html) | [CRC Catalog](https://www.routledge.com/Surrogates-Gaussian-Process-Modeling-Design-and-Optimization-for-the-Applied-Sciences/Gramacy/p/book/9780367415426)
* **Role in Thesis:** Canonical textbook for Gaussian Process regression, kernel design (Matérn 5/2, RBF), Cholesky factorization, and active learning. Cite in **Chapter 2 & 4** for `surrogate_node`.

### 15. Efficient Global Optimization of Expensive Black-Box Functions (EGO)
* **Authors:** Donald R. Jones, Matthias Schonlau, William J. Welch
* **Publication:** *Journal of Global Optimization*, 13(4), 455–492 (1998)
* **Relevance:** **Core Foundation (Essential)**
* **Link:** [https://doi.org/10.1023/A:1008306431147](https://doi.org/10.1023/A:1008306431147)
* **Role in Thesis:** Foundational paper for Kriging-based optimization. Explains how uncertainty estimation balances exploration and exploitation. Cite in **Chapter 2**.

### 16. Engineering Design via Surrogate Modelling: A Practical Guide
* **Authors:** Alexander I. J. Forrester, András Sóbester, Andy J. Keane
* **Publication:** *John Wiley & Sons*, Aerospace Series (2008)
* **Relevance:** **Core Foundation (Essential)**
* **Link:** [https://www.wiley.com/en-us/Engineering+Design+via+Surrogate+Modelling%3A+A+Practical+Guide-p-9780470060681](https://www.wiley.com/en-us/Engineering+Design+via+Surrogate+Modelling%3A+A+Practical+Guide-p-9780470060681)
* **Role in Thesis:** The standard aerospace reference book on constructing polynomial, neural network, and Kriging surrogates for aerodynamic design. Cite in **Chapter 2**.

### 17. A Python Surrogate Modeling Framework with Derivations (SMT)
* **Authors:** M. A. Bouhlel, J. T. Hwang, N. Bartoli, R. Lafage, J. Morlier, J. R. R. A. Martins
* **Publication:** *Structural and Multidisciplinary Optimization*, 59(6), 2243–2259 (2019)
* **Relevance:** **High (Formal substitution for SMT docs)**
* **Link:** [https://doi.org/10.1007/s00158-019-02200-x](https://doi.org/10.1007/s00158-019-02200-x)
* **Role in Thesis:** Benchmark reference for Gaussian Process Regression in aerospace design. Cite when discussing GP parameter fitting and validation.

### 18. EGObox: A Rust Toolbox for Efficient Global Optimization
* **Authors:** Rémi Lafage et al.
* **Publication:** *Journal of Open Source Software (JOSS)* / GitHub (2021)
* **Relevance:** **High (Implementation Reference)**
* **Link:** [https://github.com/relf/egobox](https://github.com/relf/egobox) | [Docs.rs](https://docs.rs/egobox-gp)
* **Role in Thesis:** Demonstrates production-grade Gaussian Process surrogate modeling and Mixture of Experts in Rust. Cite in **Chapter 4 (Implementation)**.

### 19. MAGPI: Multifidelity-Augmented Gaussian Process Inputs for Surrogate Modeling from Scarce Data
* **Authors:** arXiv preprint
* **Publication:** arXiv:2603.22050 (2026)
* **Relevance:** **High**
* **Link:** [https://arxiv.org/abs/2603.22050](https://arxiv.org/abs/2603.22050)
* **Role in Thesis:** Cutting-edge formulation for multi-fidelity surrogate modeling where scarce high-fidelity simulation evaluations are augmented with fast surrogate predictions. Cite in **Chapter 2**.

### 20. Gaussian Process Surrogate Model for eVTOL Propeller Aerodynamics
* **Publication:** *Proceedings of the Vertical Flight Society 81st Annual Forum* (2025/2026)
* **Relevance:** **Moderate / Contextual Application**
* **Link:** [https://proceedings.vtol.org/81/modeling-and-simulation/gaussian-process-surrogate-model-for-evtol-propeller-aerodynamics](https://proceedings.vtol.org/81/modeling-and-simulation/gaussian-process-surrogate-model-for-evtol-propeller-aerodynamics)
* **Role in Thesis:** Practical demonstration of GP surrogates replacing expensive flow computations in aerial design. Cite in **Chapter 2 (Aerospace Surrogates)**.

---

## Pillar 3: Distributed Genetic Algorithms & Island Models

### 21. Island Models Meet Rumor Spreading
* **Authors:** Tobias Friedrich, Markus Wagner, et al.
* **Publication:** *Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '17)*, pp. 1383–1390 (2017)
* **Relevance:** **Core Foundation (Essential)**
* **Link:** [https://hpi.de/friedrich/docs/publications/2017/GECCO_c.pdf](https://hpi.de/friedrich/docs/publications/2017/GECCO_c.pdf)
* **Role in Thesis:** Mathematical proof analyzing the information dissemination rate across ring topologies in Island Models. Directly justifies the ring-based peer-to-peer migration between orchestrator islands. Cite in **Chapter 3**.

### 22. Efficient and Accurate Parallel Genetic Algorithms
* **Author:** Erick Cantú-Paz
* **Publication:** *Genetic Algorithms and Evolutionary Computation*, Vol. 1, Kluwer Academic Publishers / Springer (2000)
* **Relevance:** **Core Foundation (Essential - Replaces Medium blogs)**
* **Link:** [https://doi.org/10.1007/978-1-4615-4369-5](https://doi.org/10.1007/978-1-4615-4369-5)
* **Role in Thesis:** The seminal book establishing theoretical sizing of island populations, migration frequency, and migration topologies. Replaces informal blog posts. Cite in **Chapter 2 & 3**.

### 23. A Survey of Parallel Genetic Algorithms
* **Authors:** Enrique Alba, José M. Troya
* **Publication:** *Artificial Intelligence Review*, 13(4), 255–297 (1999)
* **Relevance:** **High (Survey Foundation)**
* **Link:** [https://doi.org/10.1023/A:1006521501070](https://doi.org/10.1023/A:1006521501070)
* **Role in Thesis:** Canonical taxonomy classifying coarse-grained (island) vs. fine-grained (cellular) GAs. Cite in **Chapter 2**.

---

## Pillar 4: Aerodynamics, 3D Wing Optimization & Geometry Parameterization

### 24. AeroSandbox: A Differentiable Framework for Aircraft Design Optimization
* **Author:** Peter D. Sharpe
* **Publication:** *Master's Thesis, Massachusetts Institute of Technology (MIT)* (2021)
* **Relevance:** **Core Foundation (Essential)**
* **Link:** [https://dspace.mit.edu/handle/1721.1/139369](https://dspace.mit.edu/handle/1721.1/139369)
* **Role in Thesis:** Primary reference for the aerodynamic solver engine used in `worker/`. Explains the 3D Vortex Lattice Method (VLM) formulation and stability derivatives. Cite in **Chapter 3 & 4**.

### 25. AeroSandbox: A Framework for Aircraft Design Optimization through Automatic Differentiation
* **Authors:** Peter D. Sharpe, Andrew Ning, Mark Drela
* **Publication:** *AIAA AVIATION 2022 Forum*, Paper AIAA 2022-3860 (2022)
* **Relevance:** **Core Foundation (Essential)**
* **Link:** [https://doi.org/10.2514/6.2022-3860](https://doi.org/10.2514/6.2022-3860)
* **Role in Thesis:** Peer-reviewed AIAA publication on AeroSandbox. Explains whole-aircraft aerodynamic buildup, vortex lattice solving, and cruise trim conditions. Cite in **Chapter 3 & 4**.

### 26. Integration of the Athena Vortex Lattice (AVL) into Multivariate Design Synthesis of a Blended Wing Body Aircraft
* **Authors:** Journal of Aerospace Technology and Management / PMC
* **Publication:** *J. Aerosp. Technol. Manag.*, 15, e1323 (2023)
* **Relevance:** **High**
* **Link:** [https://pmc.ncbi.nlm.nih.gov/articles/PMC10068120/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10068120/)
* **Role in Thesis:** Validates using VLM as a fast, robust solver inside multidisciplinary optimization loops for aircraft configurations with pitch trim ($C_L = C_{L,\text{req}}$) and static margin constraints. Cite in **Chapter 2**.

### 27. OptiWing3D: A Diverse Dataset of Optimized Wing Designs
* **Authors:** Cashen Diniz, Mark Fuge
* **Publication:** *AIAA SciTech Forum* / arXiv:2512.12867 (2025)
* **Relevance:** **High**
* **Link:** [https://arxiv.org/abs/2512.12867](https://arxiv.org/abs/2512.12867)
* **Role in Thesis:** 3D aerodynamic wing design dataset highlighting 3D flow effects (tip vortex, induced drag, spanwise lift distribution) vs. 2D airfoil limitations. Validates moving beyond 2D airfoils to full 3D parametric aircraft. Cite in **Chapter 1 & 2**.

### 28. Universal Parametric Geometry Representation Method (CST)
* **Author:** Brenda M. Kulfan
* **Publication:** *Journal of Aircraft*, 45(1), 142–158 (2008)
* **Relevance:** **High (Formal substitution for MATLAB/GitHub links)**
* **Link:** [https://doi.org/10.2514/1.29958](https://doi.org/10.2514/1.29958)
* **Role in Thesis:** Definitive citation for the Class-Shape Transformation (CST) representation method. Cite when discussing airfoil parameterization choices in **Chapter 2**.

### 29. Theory of Wing Sections: Including a Summary of Airfoil Data
* **Authors:** Ira H. Abbott, Albert E. Von Doenhoff
* **Publication:** *Dover Publications*, New York (1959)
* **Relevance:** **Core Foundation (Essential)**
* **Link:** [https://archive.org/details/theoryofwingsect0000abbo](https://archive.org/details/theoryofwingsect0000abbo)
* **Role in Thesis:** Mathematical formulation for 4-digit NACA profiles (maximum camber $M$, position of maximum camber $P$, and thickness ratio $XX$), which comprise genes 8–10 in the aircraft chromosome. Cite in **Chapter 3**.

### 30. Interactive Airfoil Optimization Using Parsec Parametrization and Adjoint Method
* **Authors:** MDPI Applied Sciences
* **Publication:** *Applied Sciences*, 14(8), 3495 (2024)
* **Relevance:** **Moderate**
* **Link:** [https://www.mdpi.com/2076-3417/14/8/3495](https://www.mdpi.com/2076-3417/14/8/3495)
* **Role in Thesis:** Illustrates aerodynamic parameterization tradeoffs. Cite in **Chapter 2 (Geometric Parameterization)**.

### 31. Inverse Airfoil Design Utilizing CST Parameterization
* **Author:** D. Marshall
* **Publication:** *Cal Poly Digital Commons* (2010)
* **Relevance:** **Moderate**
* **Link:** [https://digitalcommons.calpoly.edu/context/aero_fac/article/1074/viewcontent/MarshallD_2010_InverseAirfoilDesign.pdf](https://digitalcommons.calpoly.edu/context/aero_fac/article/1074/viewcontent/MarshallD_2010_InverseAirfoilDesign.pdf)
* **Role in Thesis:** Inverse airfoil parameterization reference. Cite in **Chapter 2**.

---

## Pillar 5: Systems Engineering & Rust Distributed Infrastructure

### 32. Evaluation of Rust for Distributed Programming Compared to Go
* **Author:** Johan M. B.
* **Publication:** *Academic Comparative Study / GitHub* (2024)
* **Relevance:** **High for Architecture Justification**
* **Link:** [https://github.com/johamb/rust-distributed-programming](https://github.com/johamb/rust-distributed-programming)
* **Role in Thesis:** Provides comparative latency benchmarks between Rust and Go for distributed message passing. Directly supports Section 3 of the progress report explaining why Rust was chosen over Go to eliminate garbage collection pauses in DHT lookups. Cite in **Chapter 3**.

### 33. Envoy: A Modern Edge and Service Proxy
* **Author:** Matt Klein
* **Publication:** *ACM Queue*, 15(3), 11–22 (2017)
* **Relevance:** **High**
* **Link:** [https://queue.acm.org/detail.cfm?id=3110575](https://queue.acm.org/detail.cfm?id=3110575)
* **Role in Thesis:** Official reference for the Envoy proxy used to load-balance gRPC requests to dynamically scaled Python workers (`--scale worker=N`) using strict DNS service discovery. Cite in **Chapter 3 & 4**.

---

## Summary Mapping Table by Thesis Chapter

| Thesis Chapter | Recommended Literature to Cite |
| :--- | :--- |
| **Ch. 1: Introduction & Motivation** | OptiWing3D (#27), Forrester et al. (#16), Jin (#11) |
| **Ch. 2: Theoretical Background** | Kraska et al. (#2), Stoica et al. (#3), Gramacy (#14), Jones et al. (#15), Cantú-Paz (#22), Alba & Troya (#23), Kulfan (#28), Abbott & Von Doenhoff (#29) |
| **Ch. 3: System Architecture** | Wang et al. (LEAD) (#1), Jin et al. (Fitness Approx) (#12), Friedrich et al. (Rumor Spreading) (#21), Sharpe et al. (AeroSandbox) (#24, #25), Le Riche et al. (Local Surrogates) (#13), Rust vs Go (#32), Envoy (#33) |
| **Ch. 4: Implementation & Subsystems** | Skilling (Hilbert) (#5), Lawder & King (#6), Lv et al. (Multi-Probe) (#4), EGObox Rust (#18), Learned Metric Index Rust (#8), Bouhlel et al. (#17) |
| **Ch. 5: Experimental Evaluation & Results** | MAGPI (#19), AVL BWB (#26), Procházka (#8) |
