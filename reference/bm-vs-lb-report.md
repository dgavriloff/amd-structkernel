# BM vs LB: Why Benchmark Improvements Don't Always Translate to Leaderboard Gains

## The Eval Harness Differences

Both BM and LB use the same `_run_single_benchmark()` function in `reference/eval.py`, but with different parameters:

| | Benchmark (BM) | Leaderboard (LB) |
|---|---|---|
| `recheck` | `False` | `True` |
| `max_repeats` | 1000 | 100 |
| `max_time_ns` | 50s | 30s |
| warmup | `run_single_benchmark(tests[0], False, 100, 10e7)` — 100 reps, 100ms budget | `run_single_benchmark(tests[0], False, 100, 1e7)` — 100 reps, 10ms budget |

The `recheck=True` flag triggers three things per iteration (lines 227-246 of eval.py):

```python
if recheck:
    test.args["seed"] += 13              # 1. New seed
    data = generate_input(**test.args)    # 2. Generate entirely new input tensors
    check_copy = _clone_data(data)       # 3. Clone for correctness check

# ... kernel runs, timed ...

if recheck:
    good, message = check_implementation(check_copy, output)  # 4. Correctness check
    if not good:
        return message  # Fails the entire shape
```

## The Six Reasons BM Gains Don't Transfer to LB

### 1. Input Generation Overhead (Timed)

On LB, `generate_input()` and `_clone_data()` run every iteration. These are NOT inside the CUDA timing events, but `generate_input` may allocate GPU tensors, which can cause implicit synchronization or memory pressure that affects the kernel timing. On BM, input is generated once and reused — zero allocation pressure during timing.

**Impact**: Small but consistent. Explains the ~1-4% baseline gap seen across all kernels even when no caching is involved.

### 2. Correctness Check Overhead (Timed)

`check_implementation()` runs after every LB iteration. While the CUDA timing events only bracket `custom_kernel()`, the correctness check involves:
- Reading back the output tensor (forces GPU→CPU sync if check is on CPU)
- Computing reference output
- Comparing tensors

This doesn't directly inflate the per-kernel timing, but it adds wall-clock time that counts toward the 30s budget and 2-minute limit. With only 100 max repeats (vs 1000), fewer samples means higher variance and less averaging of outliers.

**Impact**: Higher variance on LB. A BM score of 8.91µs might show up as 9.1µs on LB purely from noise.

### 3. L2 Cache State Differences

Both BM and LB call `clear_l2_cache()` before each iteration (line 235). But on BM with fixed inputs, the data access pattern is identical every iteration — the GPU's caches, TLBs, and prefetchers learn the pattern. On LB, every iteration has new random data with different access patterns.

For kernels that are memory-bound (most of ours), this means:
- **BM**: After a few iterations, cache/TLB hit rates stabilize at their best
- **LB**: Cache/TLB hit rates are always cold-start

**Impact**: Most visible on small shapes where kernel time is dominated by memory latency. MXFP4-MM shapes at ~6µs show +4-6% LB overhead. MLA's small shapes (bs=4/kv=1k) show +7%.

### 4. Stale Buffer / Cached State Bugs (Correctness Failures)

Any state persisted across calls to `custom_kernel()` — pre-allocated output buffers, cached metadata, CUDA graphs — will work on BM (same input every time) but fail on LB (new input every time). This causes **correctness failures**, not just slower times.

**Confirmed examples from MLA**:
- **CUDA graphs** (v96): Graph captures fixed execution plan. LB replays stale computation → 118,660 mismatched elements on bs=256/kv=8192
- **Output buffer caching** (v99-v104): Pre-allocated `output = torch.empty(...)` stored in dict. ASM kernel's direct-write path leaves stale data → failures on different shapes across runs (non-deterministic)

**Why it's non-deterministic**: The specific shape that fails depends on GPU memory allocator state, fragmentation from previous shapes, and which pages happen to contain stale data. v99 failed on shape 8, v104 failed on shape 2.

**Safe to cache**: Only tensors that are truly shape-constant and never written to by kernels (e.g., `kv_indices`, `kv_last_page_len` derived purely from indptr shapes).

### 5. Warmup Differences

BM warmup: 100 reps with 100ms budget — the kernel runs many times before timing starts.
LB warmup: 100 reps with 10ms budget — 10x less time.

For kernels that rely on:
- **JIT compilation** (Triton kernels compile on first call)
- **Lazy initialization** (aiter module loading)
- **Autotuning** (Triton's autotune decorator)

The first call on LB may be partially unwarmed. BM has enough warmup to absorb all cold-start costs.

**Impact**: Primarily affects Triton-based kernels. ASM kernels (aiter) are pre-compiled and unaffected.

### 6. Statistical Sampling Differences

BM: up to 1000 repeats, stops when relative error < 0.1% or total time > 50s.
LB: up to 100 repeats, stops when relative error < 0.1% or total time > 30s.

With 10x fewer samples on LB:
- Outliers have more weight
- The mean is less stable
- A single slow iteration (cache miss, OS interrupt) shifts the mean more

For fast kernels (~6µs), 100 iterations is enough for 0.1% error. For slow kernels (~400µs), 100 iterations × 400µs = 40ms of kernel time — well within budget. So this difference is minor in practice.

**Impact**: Adds ~0.5-1% variance but doesn't systematically bias in one direction.

## Observed BM→LB Gaps Per Kernel

### MLA v105 (geomean: BM 75.4µs → LB 76.1µs, +1.0%)
Tight gap now that buffer caching is fixed. Small shapes show +2-7% from cache/TLB effects. Large shapes are within noise.

### MOE-MXFP4 v168 (geomean: BM 122.8µs → LB 124.7µs, +1.5%)
Consistent small overhead. bs=512/d=2048 shows +6% — largest shape with most memory pressure. The MoE sorting step (`moe_sorting_fwd`) allocates buffers that benefit from BM's stable memory layout.

### MXFP4-MM v224 (geomean: BM 8.7µs → LB 9.1µs, +4.1%)
Highest percentage gap. These are the fastest kernels (~6µs) so fixed overheads (L2 flush, tensor allocation) are a larger fraction. Small-M shapes are entirely memory-bound — cache warmth matters most here.

## Pre-Submission Checklist

Before proposing a change, walk through this checklist. If any answer is YES, your change will likely fail or regress on LB even if BM improves.

### Correctness (will it FAIL LB?)

- [ ] **Does `custom_kernel()` write into any tensor allocated outside the function call?**
  Global/module-level `torch.empty()`, `_cache[key] = torch.empty(...)`, etc. The ASM kernel's direct-write path leaves stale data in reused buffers when inputs change. Allocate output and intermediate buffers fresh every call.

- [ ] **Does the code use CUDA graphs?**
  `torch.cuda.CUDAGraph`, `graph.capture()`, `graph.replay()`. Graphs replay a fixed execution plan — they cannot handle new input data. This is the #1 confirmed failure mode.

- [ ] **Does the code cache anything derived from input DATA (not just shape/config)?**
  Quantized Q tensors, sorted indices from routing, pre-computed attention masks. These change when `generate_input()` runs with a new seed. Shape-derived constants (kv_indices from indptr, metadata from batch_size/kv_seq_len) are safe IF the kernel doesn't write to them.

- [ ] **Does the code assume specific seed values or data distributions?**
  Hard-coded thresholds, branch decisions based on input magnitudes, sparse optimizations that assume certain patterns. LB seeds are secret and data is random.

- [ ] **Does the code modify input tensors in-place?**
  `check_implementation()` compares the original input against reference. If you mutate the input, the correctness check sees corrupted data. The eval harness passes `_clone_data(data)` to the kernel, but if your kernel stores references to sub-tensors and mutates them, clones won't help.

### Performance (will it REGRESS on LB?)

- [ ] **Does the optimization rely on the same input running repeatedly?**
  Branch prediction warming, TLB entry stabilization, memory allocator reuse patterns. These all reset when `generate_input()` allocates new tensors with new layouts on each LB iteration.

- [ ] **Does the optimization depend on L2 cache being warm?**
  If you reorganized data access to improve cache hit rates for a SPECIFIC data layout, new random data on each iteration will have different access patterns. The improvement vanishes.

- [ ] **Is the kernel faster primarily on its first or second call?**
  Triton JIT compilation happens on first call. `@triton.autotune` runs all configs on first call. LB warmup is 10ms (vs BM's 100ms). If your kernel takes >10ms total to warm up across all shapes, LB timing includes partially-cold runs.

- [ ] **Did BM improve by more than 5% but only on small shapes (<20µs)?**
  Small shapes are most affected by cache warmth and fixed overheads. A 10% BM win on a 6µs shape might be entirely from cache effects that disappear on LB.

- [ ] **Does the code allocate large temporary buffers that persist?**
  Even if you're not caching output, large persistent allocations fragment GPU memory. On BM with fixed inputs, the allocator stabilizes. On LB with new inputs each iteration, fragmentation increases, potentially slowing allocation and causing page faults.

### Harness-Specific Traps

- [ ] **Are you optimizing for BM's specific test ordering?**
  BM runs shapes in a fixed order (as listed in the test file). If your code specializes for "shape X always follows shape Y" (e.g., keeping state from the previous shape), LB also runs the same order but with different data, so ordering assumptions about DATA are broken while ordering assumptions about SHAPES are safe.

- [ ] **Are you exploiting BM's long warmup?**
  BM warmup runs `tests[0]` for 100 iterations with 100ms budget. If your code does expensive one-time init that's amortized over BM's warmup (e.g., building lookup tables, compiling multiple kernel variants), LB's 10ms warmup may not cover it. The cost shows up in your timed runs.

- [ ] **Are you exploiting the timing window?**
  The CUDA timing events bracket only `output = custom_kernel(data)` (lines 238-240 of eval.py). Work done BEFORE the call (in module-level code, in `__init__`, at import time) is not timed. This is legitimate — precomputation of constants is fine. But if you move per-call work to import time by assuming fixed inputs, LB will re-generate inputs after your precomputation.

- [ ] **Does your code depend on `del output` (line 248) behavior?**
  The eval harness deletes the output tensor after each iteration. If your code holds a reference to the same tensor (e.g., returned a cached buffer), `del` won't actually free it. This is fine for correctness but may cause memory pressure on LB where new inputs are allocated each iteration.

### Quick Self-Test

If you can answer YES to all of these, your change is likely LB-safe:

1. If I call `custom_kernel(data)` with completely different random `data` 100 times in a row, will every output be correct?
2. If I `torch.cuda.empty_cache()` between every call, will the timing be roughly the same?
3. Does my change improve kernel compute or memory access efficiency, rather than eliminating Python/allocation/sync overhead that only exists on BM?
