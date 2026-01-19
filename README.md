# REMatch-SOS

## About REMatch/SOS

**REMatch + SOS** is an **unsupervised, structure-first screening** method for large pools of candidate atomic structures.
Instead of learning energies/forces from labels, it **measures structural similarity** and selects a **diverse, informative subset**
for expensive downstream calculations (e.g., geometry relaxations, DFT, dataset curation).

### What it does
Given a large set of initial configurations, REMatch/SOS:
- encodes each structure with **SOAP** descriptors (optionally compressed),
- computes pairwise structural similarity via the **REMatch kernel**,
- converts similarity to distances and assigns each structure an **outlier probability** using **Stochastic Outlier Selection (SOS)**,
- retains **boundary structures** (neither dense-cluster centers nor extreme outliers) as a representative subset.

In our published testbed, this reduces **10,000 initial configurations per case to ~30% (~3,000)** for full relaxation while
retaining the key low-energy landscape.

### How it differs from supervised ML potentials (e.g., MACE)
Supervised interatomic potentials (e.g., **MACE**) are trained to predict **energies/forces** from labeled data, making them ideal
for:
- quickly removing **unphysical / extremely high-energy** structures,
- fast ranking, MD, or local relaxation.

**REMatch/SOS is complementary**: it does **not** require E/F labels and does not attempt to predict energies.
Its goal is **redundancy reduction and coverage**: after you remove pathological structures (with MACE or any fast surrogate),
REMatch/SOS helps you keep structures that are **structurally distinct** so you spend compute on **new information**, not repeats.

### Why it is interpretable (structure-first)
Selections can be explained directly in terms of structure:
- each structure has an **outlier probability** (SOS),
- retained points are **boundary / informative configurations** in the similarity space,
- you can trace *why* a structure is kept or discarded via its similarities to neighbors (kernel distances) and its SOS score.

### Typical use cases
- Pre-screening before expensive relaxations (DFT / high-level methods)
- Building diverse training sets for ML potentials / active learning
- Removing near-duplicates from large structure pools (MC/GA/random searches)



Machine Learning Accelerated Structure Prediction for Supported Metal Nanoclusters
10.1103/PhysRevMaterials.9.033801
