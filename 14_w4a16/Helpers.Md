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
