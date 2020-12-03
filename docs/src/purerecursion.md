
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

!!! note "Alternative Path"
      For a general recursion  ${\bm k} = {\bm k}' \cup {\bm k}''$ this looks
      quite difficult. It may be preferrable to construct an alternative, much
      larger DAG and allow only ${\bm k} = (k_1, \dots, k_{\nu-1}) \cup (k_\nu,)$
      recursions. In this case, the next steps are much easier, but we would pay
      for it through the cost of the much larger DAG.

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
$\sum_{p \text{matches}}$ in terms of $A_{\bm k}$ with $\mathcal{A}_{\bm k}$ with ${\rm len}({\bm k}) < \nu$.
