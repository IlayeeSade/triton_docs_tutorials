Useful comment for delete cache of compiled CUDA kernels
```
rm -rf ~/.cache/torch_extensions/
```
The size of a bus is 128 bytes, which means when we load from memory we want optimally to load 128 bytes for a warp.

We generally prefer 128-256 threads per thread-block because then we can probably still reach 100 occupany,
and we have 16-8 blocks on the SM so if we wait for memory operations other blocks are in the queue and therefore
calculate stuff so that we did not waste any time. (L1/L2 Caches)
- syncthreads takes longer for larger blocks
- there is a physical limit for small blocks, 32
- larger blocks might require a lot of the SM's memory, thus will not load other blocks even though they "fit"



# CUDA Memory Hierarchy

| Memory Type | Scope | Lifetime | Size (Typical) | Latency | Bandwidth | Managed By | Shared Between | Typical Use |
|-------------|------|----------|---------------|---------|-----------|------------|----------------|-------------|
| Registers | Thread | Kernel | ~64–256 KB per SM (split across threads) | ~1 cycle | Extremely high | Compiler | No | Local variables, accumulators |
| Shared Memory | Thread Block | Kernel | 64–164 KB per SM (Ampere configurable with L1) | ~10–20 cycles | Very high | Programmer | Threads in same block | Tiling, data reuse |
| L1 Cache | SM | Kernel | ~128 KB per SM (Ampere shared with shared memory) | ~20–30 cycles | High | Hardware | All threads on same SM | Cached global loads |
| Read-only Cache (`__ldg`) | SM | Kernel | Uses L1 read-only path | ~20–30 cycles | High | Hardware | All threads on same SM | Weights, activations |
| L2 Cache | GPU | Kernel | 4–6 MB (Ampere ~4MB for 3070) | ~100–200 cycles | High | Hardware | All SMs | Global data reuse |
| Global Memory (DRAM) | GPU | Application | 8–24 GB typical | ~400–800 cycles | High but high latency | Programmer | All threads | Main tensors |
| Constant Memory | GPU | Application | 64 KB | ~1–2 cycles (broadcast) | Very high for broadcast | Programmer | All threads | Small read-only data |
| Texture Memory | GPU | Kernel | Uses texture cache | ~100 cycles | Optimized for spatial locality | Programmer | All threads | Image data |
| Local Memory | Thread | Kernel | In global memory | Same as global | Same as global | Compiler | No | Register spill |

---

# Cache Sharing Behavior

| Resource | Shared By |
|--------|-----------|
| Registers | Single thread |
| Shared memory | Threads in same block |
| L1 cache | All warps running on same SM |
| L2 cache | Entire GPU |
| Global memory | Entire GPU |

---

# Typical Access Path


Registers
↓
Shared Memory (optional)
↓
L1 Cache / Read-only Cache
↓
L2 Cache
↓
Global Memory (DRAM)


---

# Key Performance Rules

| Rule | Reason |
|-----|------|
| Prefer registers | Fastest memory |
| Reuse via shared memory | Avoid repeated global loads |
| Coalesce global memory accesses | Maximize bandwidth |
| Use `__ldg` for read-only data | Uses read-only cache path |
| Minimize global memory accesses | DRAM latency is large |
| Tile computations | Increase data reuse |

# Achieved Occupancy

Achieved occupancy might be lower than theoretical because it counts activate number of warps which are not stalled. This practically means that warp stall causes discrepencies.

Branching is pretty cheap.


| Layer | What it Represents | Unit Size | Who Controls It | What It Means | Example |
|------|------|------|------|------|------|
| Thread Instruction Load | The amount of data a **single thread loads in one instruction** | e.g. 32-bit, 64-bit, 128-bit | Your code / compiler | Determines how many bytes a thread requests at once | `float`, `float2`, `float4`, `uint4` |
| Warp Memory Transaction | The **actual memory transfer done by the GPU memory system** | Usually **128 bytes** | GPU hardware | The warp's requests are merged and served in 128-byte segments | A warp loading 32×4 bytes |

---

# Warp Coalescing Dilemma

| Scenario | Thread Access Pattern | Warp Address Pattern | Transactions Needed | Performance | Explanation |
|------|------|------|------|------|------|
| Bad Case | Each thread loads 4 bytes but from unrelated locations | Addresses are scattered across memory | Up to **32 transactions** | ❌ Very Slow | Each thread touches a different 128-byte segment, so the GPU must fetch many memory lines |
| Good Case | Each thread loads 4 bytes that are unrelated within the thread | But **all 32 threads together cover a contiguous 128-byte region** | **1 transaction** | ✅ Fast | The warp accesses a single contiguous cache line, so the GPU serves everything with one memory fetch |

---

# Key Insight

The GPU optimizes **warp memory access**, not individual thread locality.

So the important rule is:


Make the warp access contiguous memory.


It **does not matter** if a single thread's accesses are internally contiguous, as long as:


the combined warp request fits into few 128-byte segments.


---

# Practical Rule for CUDA Kernels

Best pattern:


thread_i → base + i * element_size


Example:


thread0 → A[0]
thread1 → A[1]
thread2 → A[2]
...
thread31 → A[31]


Worst pattern:


thread_i → base + i * large_stride


Example:


thread0 → A[0]
thread1 → A[1024]
thread2 → A[2048]
...


---

# Short Summary

| Question | Answer |
|------|------|
| Should each thread access contiguous memory? | Not necessary |
| Should the **warp** access contiguous memory? | **Yes (critical)** |
| What determines memory performance? | Number of **128-byte memory transactions** |


Cloascing does not happen over threads blocks, but it might improve L2 hits.
Memory coalescing happens per warp, so contiguity between warps generally does not improve coalescing. 
