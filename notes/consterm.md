
For a `SymmetricBasis`, we may hope it to have contant terms which may complete the basis further. The way we represent those constant terms in a `SymmetricBasis` is that we give it zero-lengthed `spec` as well as its initial coupling coefficients manually by defining
```
coco_init(φ<:AbstractProperty)
```
The remaining shall be exactly the same. We should point out that not all properties can have constant term(s).

e.g., If an `EuclideanVector` $F$ is a constant, then it should satisfy 
$$
    F = QF, \forall Q\in O(3).
$$
From which we must have $F = \bm 0$.

Analogously, assume $H$ is a constant belonged to `SphericalTensor`, then
$$
    D^{L_1}(Q)^\ast H D^{L_2}(Q) = H, \forall Q\in O(3).
$$
We may write $H = \sum_{i,j}^{2L_1+1,2L_2+1} h_{ij}E^{ij}$, then by integraling both sides over $O(3)$, we have 
$$
    \sum_{i,j}^{2L_1+1,2L_2+1}\int_{O(3)} D^{L_1}(Q)^\ast E^{ij}D^{L_2}(Q) dQ h_{ij} = \sum_{i,j}^{2L_1+1,2L_2+1} h_{ij}E^{ij}
$$
As a result, we see that the constant terms can only appear when $L_1 == L_2$, which are $cI(2L_1+1)$ by comparing both sides element-wised and using the orthogonality of Wigner-D matrices.

More examples is to be added...