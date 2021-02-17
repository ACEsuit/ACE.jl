
### Pure Basis Recursion

The idea is to exploit the DAG representation of the pi-ACE basis $A_{\bf k}$ to recursively construct the pure basis $\mathcal{A}_{\bm k}$.

#### 1-correlations (2-body)

For the 1-correlations, there is nothing to do since
```math

   A_k = \mathcal{A}_k \qquad \forall k
```

#### 2-correlations (3-body)

If ${\bm k} = (k_1, k_2)$ then the connection ${\bm k} = (k_1,) \cup (k_2,)$ is already the DAG. We can now write
```math
\begin{aligned}
   {A}_{k_1 k_2}
   &=
   \sum_{j_1, j_2} \phi_{k_1}({\bm r}_{j_1}) \phi_{k_2}({\bm r}_{j_2}) \\
   &=
   \sum_{j_1 \neq j_2} \phi_{k_1}({\bm r}_{j_1}) \phi_{k_2}({\bm r}_{j_2})
   + \sum_{j_1} \phi_{k_1}({\bm r}_{j_1}) \phi_{k_2}({\bm r}_{j_1}),
\end{aligned}
```
or, conversely,
```math
   \mathcal{A}_{k_1 k_2}
   =
   {A}_{k_1 k_2} - \sum_{j_1} \phi_{k_1}({\bm r}_{j_1}) \phi_{k_2}({\bm r}_{j_1}).
```
Next we observe that the product of polynomials $\phi_{k_1} \phi_{k_2}$ is again a polynomial, i.e.,
```math
   \phi_{k_1} \phi_{k_2}
   = \sum_K P_{k_1 k_2}^K \phi_K
```
hence, we can write
```math
   \mathcal{A}_{k_1 k_2}
   =
   {A}_{k_1 k_2} - \sum_{K} P_{k_1 k_2}^K A_K
```
The computation of the product coefficients $P_{k_1 k_2}^K$ will be discussed  [Products of Polynomials](@ref).

#### General correlations

For general ${\bm k} = (k_1, \dots, k_\nu)$, the DAG provides a recursion
```math
   {\bm k} = {\bm k}' \cup {\bm k}''
```
where
```math
   {\bm k}' = (k_1, \dots, k_{\nu'}), \qquad
   {\bm k}'' = (k_{\nu'+1}, \dots, k_\nu),
```
with $\nu'' = \nu - \nu'$.

Since we already have access to $\mathcal{A}_{\bm k'}, \mathcal{A}_{\bm k''}$ we can start from those objects,
```math
   \mathcal{A}_{\bm k'}, \mathcal{A}_{\bm k''}
   =
   \sum_{j_1 \neq \dots \neq j_{\nu'}}
   \sum_{j_{\nu'} \neq \cdots \neq j_{\nu}}
   \prod_{\alpha = 1}^\nu \phi_{k_\alpha}(\bm r_{j_\alpha})
```
The challenge now is to convert this into a $\sum_{j_1 \neq \cdots \neq j_\nu}$.
For a general recursion  ${\bm k} = {\bm k}' \cup {\bm k}''$ this looks
quite difficult.

Let us assume that $\nu' \leq \nu''$ then we can write
```math
   \sum_{j_1 \neq \dots \neq j_{\nu'}}
   \sum_{j_{\nu'} \neq \cdots \neq j_{\nu}}
   =
   \sum_{0 \text{~matches}} +
   \sum_{1 \text{~matches}}
   + \cdots +
   \sum_{\nu \text{~matches}}
```
where
```math
   \sum_{p \text{~matches}}
   =
   \sum_{\substack{
         j_1 \neq \cdots \neq j_{\nu'} \\
         j_{\nu'+1} \neq \cdots \neq j_{\nu} \\
         \# \{ j_1, \dots, j_{\nu'} \} \cap \{j_{\nu'+1}, \dots, j_\nu\} = p}}
```
Clearly,
```math
   \sum_{0 \text{~matches}}
   =
   \sum_{j_1 \neq \cdots \neq j_\nu}
```
i.e. this is the term we want to keep. It remains to express
$\sum_{p \text{matches}}$ in terms of $A_{\bm k}$ with $\mathcal{A}_{\bm k}$ with ${\rm len}({\bm k}) < \nu$. It appears that for $p$ there are
```math
   \binom{\nu'}{p}
      \cdot
   \binom{\nu''}{p}
```
sums to evaluate. This cost clearly explodes rapidly with increasing body-order.

#### Simplified recursion

To control this cost we can replace an optimized DAG with a much simpler DAG that contains only decompositions ${\bm k'} = {\bm k} \cup (k_{\nu+1},)$ (note the modified notation to avoid clutter below). In this case, the term we need to manipulate is
```math
\begin{aligned}
   \mathcal{A}_{\bm k} A_{k_{\nu+1}}
   &=
   \sum_{j_1 \neq \cdots \neq j_\nu} \sum_{j_{\nu+1}} \prod_{\alpha = 1}^{\nu+1} \phi_{k_\alpha}(r_{j_\alpha}) \\
   &=
   \sum_{j_1 \neq \cdots \neq j_\nu \neq j_{\nu+1}} \prod_{\alpha = 1}^{\nu+1} \phi_{k_\alpha}(r_{j_\alpha})
   +
   \sum_{\beta = 1}^{\nu} \sum_{j_1 \neq \cdots \neq j_\nu}
      \phi_{k_{\nu+1}}(r_{j_\beta}) \prod_{\alpha = 1}^{\nu} \phi_{k_\alpha}(r_{j_\alpha}).
\end{aligned}
```

We can now insert the expression for [Products of Polynomials](@ref) to write
```math
   \phi_{k_\beta}(r_{j_\beta}) \phi_{k_{\nu+1}}(r_{j_\beta})
   =
   \sum_{K} P_{k_\beta k_{\nu+1}}^K
   \phi_K(r_{j_\beta})
```
and obtain
```math
   \mathcal{A}_{\bm k} A_{k_{\nu+1}}
   =
   \mathcal{A}_{({\bm k}, k_{\nu+1})}
   +
   \sum_{\beta = 1}^\nu
   \sum_{K} P_{k_\beta k_{\nu+1}}^K
   \mathcal{A}_{{\bm k}[\beta]}
```
where
```math
   {\bm k}[\beta] := (k_1, \dots, k_{\beta-1}, K, k_{\beta+1}, \dots, k_\nu).
```
With this definition we have obtained a recursive expression the ``pure'' $\mathcal{A}$  basis,
```math
   \mathcal{A}_{({\bm k}, k_{\nu+1})}
   =
   \mathcal{A}_{\bm k} A_{k_{\nu+1}}
   -
   \sum_{\beta = 1}^\nu
   \sum_{K} P_{k_\beta k_{\nu+1}}^K
   \mathcal{A}_{{\bm k}[\beta]}
```
