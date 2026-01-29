
---

# Modular Arithmetic Tasks

This directory contains the implementation and ablation experiments for
**modular arithmetic tasks** used in the paper.

---

## Basic Experiments (Baseline vs. R2G)

We provide demo scripts to reproduce the core modular addition experiments
under both the **baseline loss** and **R2G Loss**.

### Demo Commands

```bash
python transformer.py 26650
```

Runs modular addition under the **baseline loss** with **26,650** training
samples.

```bash
python loss2.py 26650
```
Runs the same modular addition task under **R2G Loss** with **26,650**
training samples.

To run loss2.py, an additional hook is required to expose the internal
Query (Q), Key (K), and Value (V) representations from the
Transformer attention module.

Specifically, please modify modeling_openai.py in the corresponding
transformers library by adding the following lines to the forward
function:
```python
def forward(self, x, attention_mask=None, head_mask=None, output_attentions=False):
    x = self.c_attn(x)
    query, key, value = x.split(self.split_size, dim=2)

    # Hook for R2G 
    self.last_q = query
    self.last_k = key
    self.last_v = value
```

---

## Experiments with Different Moduli and Operations

To reproduce experiments with **different moduli** or **different modular
operations** (e.g., subtraction or multiplication), modify the relevant
variables in the files under the `one-layer-openai-gpt` directory
accordingly.

---

## Ablation on Q, K, and V Vectors

To perform ablation studies on different attention components:

* By default, R2G Loss is applied to **Value (V)** vectors.
* To apply R2G Loss to **Query (Q)** or **Key (K)** vectors, simply replace
  the `v` vectors with `q` or `k` in `loss2.py`.

This ablation is used to verify that R2G is most effective when applied to
the Value space.

---

## Norm-only and Angular-only Ablation Experiments

To reproduce the **Norm-only Loss** and **Angular-only Loss** ablations,
replace the R2G loss computation in the training and validation sections of
`loss2.py` with the following implementations.

---

### Norm-only Loss

This variant encourages embeddings to grow in magnitude without enforcing
any angular separation.

```python
# Compute L2 norms of raw embeddings
norm0 = torch.norm(pos0_train, dim=-1)
norm2 = torch.norm(pos2_train, dim=-1)
norm6 = torch.norm(pos6_train, dim=-1)

# Objective: encourage larger norms (smaller inverse loss)
total_d_train = torch.sum(norm0) + torch.sum(norm2) + torch.sum(norm6)
dis_loss_train = 10000 / total_d_train  # 10000: lambda
```

---

### Angular-only Loss

This variant isolates angular separation by normalizing embeddings to unit
norm, ensuring that distances reflect only directional differences.

```python
# Key modification: normalize embeddings to remove norm effects
pos0_val = F.normalize(pos0_val, p=2, dim=-1)
pos2_val = F.normalize(pos2_val, p=2, dim=-1)
pos6_val = F.normalize(pos6_val, p=2, dim=-1)

# Pairwise distance computation via random permutation
random_indices_val = torch.randperm(pos0_val.size(0))
dis0_val = torch.norm(pos0_val - pos0_val[random_indices_val], dim=1)
dis2_val = torch.norm(pos2_val - pos2_val[random_indices_val], dim=1)
dis6_val = torch.norm(pos6_val - pos6_val[random_indices_val], dim=1)

total_d_val += (
    torch.sum(dis0_val).item()
    + torch.sum(dis2_val).item()
    + torch.sum(dis6_val).item()
)

dis_loss_val = 10000 / total_d_val
```

---

