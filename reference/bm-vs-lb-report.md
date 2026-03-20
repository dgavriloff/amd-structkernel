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

Answer NO to all of these or your change will fail/regress on LB:

### Will it break?
- [ ] Any tensor written to by the kernel cached across calls? (output buffers, intermediates)
- [ ] CUDA graphs?
- [ ] Caching anything derived from input data (not shape/config)?
- [ ] Mutating input tensors in-place?

### Will it regress?
- [ ] Improvement depends on same data running repeatedly? (cache warmth, TLB, allocator patterns)
- [ ] Big BM win (>5%) only on small shapes (<20µs)? Likely cache effects that vanish on LB.
- [ ] Relies on 100ms warmup? LB gets 10ms.

### Quick self-test
Call `custom_kernel()` 100 times with different random data each time. Is every output correct and roughly the same speed?
