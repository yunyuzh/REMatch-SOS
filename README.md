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

REMatch/SOS is **unsupervised** and **structure-first**: it removes redundancy by measuring *structural similarity*
(descriptor + kernel), not by learning energies/forces.

### Energy-landscape assumptions (paper-motivated)
This workflow is motivated by three common (system-dependent) assumptions about the energy landscape:
1. **Similar initial structures tend to relax to similar local minima.**
2. **Global minima are typically surrounded by structurally (and often energetically) similar local minima.**
3. **As system size increases, the proportion of distinct low-energy structures decreases.**

### Practical requirements
- **Comparable inputs**: structures should represent the same “type of object” (same chemistry / consistent definition).
  If you mix qualitatively different systems, **split into groups** and run selection per group.
- **Descriptor fidelity**: your descriptor (e.g., SOAP) must capture the structural differences you care about.
- **Redundancy exists**: the method is most useful when the pool contains many near-duplicates.
- **Prefilter recommended**: remove unphysical / extremely high-energy structures first (e.g., via MACE or simple heuristics).

### Recommended scope (works best when)
- Large candidate pools from MC/GA/random searches where you want a **diverse subset** before expensive relaxation/DFT.
- Dataset curation / active learning where you want **coverage**, not only “top-ranked by energy”.

### Limitations / when to be careful
- If key physics is not represented in the descriptor/kernel, “diversity” in similarity space may not match what you need.
- Kernel-based similarity can be heavy at scale (pairwise growth); use batching/approximation strategies for large runs.

### ⚠️ Validate before large-scale runs
Before running on very large datasets, do a small pilot (e.g., 500–2000 structures) to calibrate:
- descriptor/kernel settings (do neighborhoods look sensible?),
- SOS threshold range vs. subset size,
- (if energies are available) whether the selected subset preserves low-energy regions and the overall distribution.


