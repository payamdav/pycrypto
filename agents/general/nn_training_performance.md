# Neural-Net Training Performance Techniques

> **Placement rationale:** This file lives in `agents/general/` because it
> defines a repository-wide **convention** that every agent must apply when
> training or evaluating neural networks (per `agents/general/paths_and_files.md`:
> "Any document that defines a general rule, convention, or specification …
> belongs here"). It is not a single algorithm/idea spec, so it does not belong
> in `agents/ideas/`.

This document instructs coding agents to apply a set of **performance**
techniques whenever they train or evaluate neural nets — but **only** the ones
that do not change the learning result. The golden rule:

> **Make it faster, not different.** A free speedup changes *how* the hardware
> is scheduled, never *what* math the model performs. If a change alters the
> sequence of weight updates, the loss, the data seen per step, or the
> initialization, it is **tuning**, not a speedup — treat it deliberately.

---

## Free speedups (same statistical result)

### 1. Keep small datasets resident on the GPU

If the dataset fits in VRAM (most engineered feature/label tables do), move it
to the device **once** and slice batches by index. Avoid `DataLoader` /
`TensorDataset` worker overhead and per-batch host→device copies — they add
latency without touching the math.

```python
# transfer ONCE
x = torch.from_numpy(x_np.astype(np.float32)).to(device)
y = torch.from_numpy(y_np.astype(np.float32)).to(device)

for epoch in range(max_epochs):
    perm = torch.randperm(x.shape[0], device=device)   # same reshuffle as DataLoader(shuffle=True)
    for s in range(0, x.shape[0], batch_size):
        idx = perm[s:s + batch_size]
        loss = criterion(model(x[idx]), y[idx])
        ...
```

> Preserve the **full per-epoch random permutation** so the batch composition
> matches `DataLoader(shuffle=True)`. Same data per step ⇒ same result.

### 2. Minimize host↔device syncs

Every `.item()`, `.cpu()`, or `print` of a tensor forces a CUDA sync that
stalls the pipeline. Accumulate metrics in an **on-device** tensor and call
`.item()` once per epoch — for logging only, never inside the loss/gradient
path.

```python
running = torch.zeros((), device=device)   # on-device accumulator
for s in range(0, n, batch_size):
    loss = criterion(model(x[idx]), y[idx])
    loss.backward(); opt.step(); opt.zero_grad()
    running += loss.detach() * idx.numel()  # no sync
epoch_loss = (running / n).item()           # one sync per epoch
```

### 3. `torch.backends.cudnn.benchmark = True`

When input shapes are fixed across iterations (constant sequence length and
feature count, last partial batch aside), let cuDNN autotune its kernels once.
Set it at startup. The chosen kernels are mathematically equivalent — this is a
statistically-same-result speedup.

```python
torch.backends.cudnn.benchmark = True   # fixed shapes only
```

### 4. Fill the GPU when training MANY small independent models

Small models are **launch-bound**: each kernel finishes faster than the CPU can
queue the next, so a single model leaves most of the GPU idle. A *bigger* GPU
does **not** help — you must add concurrency. Run several independent models at
once on the one device:

- **CUDA streams (in-process):** give each model its own
  `torch.cuda.Stream` and interleave their steps; the GPU overlaps their
  kernels. Synchronize at wave boundaries. Each model keeps its own optimizer /
  scheduler / state, so per-model results are unchanged.
- **`torch.func.vmap` / functorch ensembling:** vectorize identical
  architectures into one batched forward when truly homogeneous.
- **Multiprocessing or CUDA MPS:** separate processes sharing the GPU.

```python
streams = [torch.cuda.Stream(device=device) for _ in wave]
for i, model_ctx in enumerate(wave):
    with torch.cuda.stream(streams[i]):
        run_one_epoch(model_ctx)   # independent optimizer/state per model
torch.cuda.synchronize()           # boundary
```

> Streams only overlap independent kernels **in time**; they never mix
> gradients or state between models. Per-model math is identical to running
> them one-by-one.

### 5. Always separate and report data-load time vs compute time

Time data loading, host→device transfer, and training **separately** and print
each. Conflating them hides where the bottleneck is and leads to "fixing" the
wrong thing (e.g. buying GPU horsepower for an I/O-bound load).

---

## Result-changing choices (NOT free — treat as deliberate tuning)

These **do** change the learning result. Never apply them as a silent
"speedup"; only change them as an intentional, documented experiment:

- **Batch size** — changes the number and noise of gradient updates per epoch,
  the effective learning rate, and BN/regularization behavior. Bigger batches
  are *faster per epoch* but produce a **different** model. **Do not change
  batch size to go faster.**
- **Seeds** — different initialization / shuffle order ⇒ different weights.
- **Model architecture** (layers, units, dropout) — different model entirely.
- **Optimizer / LR / scheduler / weight decay / loss** — different optimization
  trajectory.
- **Mixed precision (AMP/fp16/bf16)** — usually a near-free speedup, but it
  *can* shift results slightly via reduced precision; validate before treating
  it as free, and disable if it perturbs metrics.
- **Dropping the last partial batch / changing the purge gap / split** —
  changes the data the model trains and is evaluated on.

---

## Determinism note

GPU training is **already non-deterministic across hardware** (atomic reduction
order, library versions, kernel selection). The free speedups above keep you in
the **same statistical regime** — they do not add divergence beyond what GPU
training already has. When you need bit-exact reproducibility, that is a
separate, slower mode (`torch.use_deterministic_algorithms(True)`,
`cudnn.benchmark = False`), and it is itself a deliberate trade-off — not a
performance technique.

---

## Checklist for agents

When you write or modify NN training/eval code, confirm:

1. [ ] Dataset resident on GPU if it fits; batches sliced by index (no per-batch copy).
2. [ ] Metrics accumulated on-device; `.item()` once per epoch.
3. [ ] `cudnn.benchmark = True` if input shapes are fixed.
4. [ ] Many small models ⇒ concurrency (streams / vmap / processes), not a bigger GPU.
5. [ ] Data-load, transfer, and compute times reported separately.
6. [ ] Batch size / seed / architecture / optimizer **unchanged** unless tuning is the explicit goal.
