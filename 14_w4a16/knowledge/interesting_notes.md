1. Shared memory GEMM vs GEMV vs Rereading minus overhead
2. Taking as input types who are different and casting
3. Pragma unroll, force inline
4. Excess calculations for the same groups
5. Divison is very slow
6. Butterfly reduction
7. Reusing as much as you can, but also consider structual requirements and utilizing them for the compression mechanism and method in general.
8. Banks, 32 banks for warp, notice the access different banks (Leap-frogging)
9. Leap-Frogging.
10. Coalecsing.
11. Software-pipelining.
12. ILP