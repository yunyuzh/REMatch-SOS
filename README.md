# REMatch-SOS

Machine Learning Accelerated Structure Prediction for Supported Metal Nanoclusters
10.1103/PhysRevMaterials.9.033801

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

### How it differs from supervised ML potentials
Supervised interatomic potentials (e.g., **MACE**) are trained to predict **energies/forces** from labeled data, making them ideal
for:
- quickly removing **unphysical / extremely high-energy** structures,
- fast ranking, MD, or local relaxation.

**REMatch/SOS is complementary**: it does **not** require E/F labels and does not attempt to predict energies.
Its goal is **redundancy reduction and coverage**: after you remove pathological structures (with MACE or any fast surrogate),
REMatch/SOS helps you keep structures that are **structurally distinct** so you spend compute on **new information**, not repeats.

### Typical use cases
- Pre-screening before expensive relaxations (DFT / high-level methods)
- Building diverse training sets for ML potentials / active learning
- Removing near-duplicates from large structure pools (MC/GA/random searches)

## Assumptions & scope

REMatch/SOS is **unsupervised** and **structure-first**: it assumes that *structural similarity* (as defined by your descriptor + kernel)
is a meaningful proxy for redundancy / diversity in your dataset.

### Core assumptions
- **Comparable structures**: all inputs represent the *same “kind” of object* (e.g., same chemical system / composition and a consistent definition of the structure).
  If you mix qualitatively different systems, **split into groups** and run selection per group.
- **Descriptor fidelity**: the chosen descriptor (e.g., SOAP) captures the structural features you care about (local coordination, bonding motifs, etc.).
- **Redundancy exists**: the pool contains many near-duplicates; selection is beneficial only when redundancy is non-trivial.
- **Goal is coverage, not direct ranking**: the method prioritizes **representative diversity**. It does not “learn” energies/forces and is not a replacement for a potential.

### Recommended scope (works best when)
- You have **large candidate pools** from MC/GA/random searches and need to remove near-duplicates before expensive relaxation/DFT.
- You are curating a **diverse subset** for downstream tasks (DFT labeling, ML training, active learning).
- Your dominant variability is **geometric / structural** (rather than, e.g., electronic state changes not encoded in the descriptor).

### Limitations / when to be careful
- If your descriptor misses the “physics that matters” for your problem, selection may not preserve what you care about.
- **Pathological / extremely high-energy structures** can distort similarity space.  
  **Tip:** prefilter with a fast surrogate potential (e.g., MACE) or simple heuristics before running REMatch/SOS.
- Kernel-based methods can be heavy at scale (pairwise similarities grow quickly with dataset size); consider approximation / batching strategies for large runs.

### ⚠️ Please validate before large-scale use
For a *new system*, do a small pilot study first (e.g., a few hundred to a few thousand structures) to check that:
- your chosen descriptor/kernel settings produce sensible neighborhoods,
- your SOS thresholds yield the desired subset size,
- (if energies are available) the selected subset still captures low-energy regions and preserves the overall distribution.

A simple threshold sensitivity scan is often enough to calibrate robust defaults for your dataset.


