
### What is Pure ACE

The standard ACE basis is given by the $O(3)$-symmetrised $\nu$-correlations. Ignoring the $O(3)$-symmetry for a moment, we are talking about
```math
A_{\bf k} = \prod_{\alpha} A_{k_\alpha}
```
where
```math
   A_k = \sum_j \phi_k({\bm r}_j).
```
Another way to write this is,
```math
A_{\bf k}
=
\sum_{j_1, j_2, \dots, j_\nu}
   \prod_{\alpha = 1}^\nu
   \phi_{k_\alpha}({\bm r}_{j_\alpha}),
```
which shows the potentially unwelcome self-interaction terms.

On the other hand, the "naive" summetrised $\nu$-order interaction is given by
```math
   \mathcal{A}_{\bm k} =
   \frac{1}{\nu!} \sum_{\sigma \in S_\nu} \phi_{\bm k} \circ \sigma
```
We call this the "pure" ACE basis because it contains no self-interactions;
indeed, another way to write it is
```math
   \mathcal{A}_{\bm k}
   =
   \sum_{j_1 < \cdots < j_\nu}
   \prod_{\alpha = 1}^\nu
      \phi_{k_\alpha}( {\bm r}_{j_\alpha} )
```

Maybe we shouldn't care about this at all, but this is far from clear. First it
is chemically more intuitive to work with this basis and maybe this has some
advantages in constructing or sparsifying good models. But from a purely
numerical perspective there is the problem that the standard ACE basis is
ill-conditioned for large body-orders.  However, if the $\phi_{\bm k}$ are orthogonal, then this orthogonality is inherited by the $\mathcal{B}_{\bm k}$. (TODO: some subtle points - insert details)

Having an orthogonal basis has clear theoretical advantages and it gives us the opportunity to explore whether these translate also into practical advantages.
