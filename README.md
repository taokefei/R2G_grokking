## The Geometric Origin of Grokking:

### Accelerating Generalization via Active Structural Reorganization

This repository provides the implementation of **R2G (Repel-to-Grokking) Loss**, proposed in *The Geometric Origin of Grokking*, and the corresponding experimental code for both **modular arithmetic tasks** and the **tense-inflection task**.

R2G enhances generalization by explicitly enforcing structural repulsion among token embeddings. The core objective is defined as:

$$
L_{\mathrm{R2G}} = \frac{1}{\sum_{i=1}^{n} |v_i - v'_i|_2},
$$

where ( $v'_i$ ) denotes a randomly permuted counterpart of ( $v_i$ ).

---

## Environment Setup

* Python 3.12
* Required packages are listed in the project code files.

---

## Modular Arithmetic Tasks

We provide the implementation for **modular addition with modulus ( $P = 500$ )**.
Other modular arithmetic tasks reported in the paper can be reproduced by modifying the arithmetic operator or task parameters accordingly.

### Running the Experiments

Navigate to the *modular arithmetic task* directory.

**Demo commands:**

```bash
python transformer.py 26650
```

Runs modular addition under the **baseline loss** with 26,650 training samples.

```bash
python loss2.py 26650
```

Runs the same task under **R2G Loss** with 26,650 training samples.

---

## Tense-Inflection Task

This experiment evaluates the effectiveness of R2G Loss on a **non-arithmetic linguistic task**, testing its generality across domains.

We build upon the codebase from:

> *Grokking of Hierarchical Structure in Vanilla Transformers*
> Murty et al., 2023

### Running the Experiment

Navigate to the *tense-inflection task* directory.

**Demo command:**

```bash
python train_transformers.py \
  --dataset tense \
  --save_dir ./10000_r2g_0.1 \
  --traindata 10000 \
  --alpha 0.1 \
  --callback
```

### Argument Description

* `--dataset tense`
  Specifies the tense-inflection task.

* `--save_dir ./10000_r2g_0.1`
  Directory for saving trained models and logs.

* `--traindata 10000`
  Number of training samples.

* `--alpha 0.1`
  Weight of the R2G Loss.
  If `--alpha` is not specified, the model is trained using the **baseline loss**.



