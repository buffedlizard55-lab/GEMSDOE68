# Why the emission budget is a design choice, not a holdout optimum

The DTI denominator rises by exactly `alpha = 0.2` for **every** unit of emitted mass,
wherever it lands, because

```
1/DTI = alpha + alpha*(F/T) + beta*(K/T)      alpha=0.2, beta=0.8
```

(`src/gems68/metric.py`, proved in `tests/test_metric.py::test_identity_one_over_dti`).
A dot therefore pays for itself iff its realised kernel weight `k` exceeds `alpha * DTI`;
at DTI 0.2778 that bar is 0.05556, and the only lattice offsets inside it are
`d <= 2.828 px` = 283 m.

Two facts make the local holdout unusable for choosing the budget:

1. The organizer masks pixels that sit on already-mapped faults out of the scored domain.
   Our holdout's truth **is** that catalogue, so the holdout rewards mass on and beside the
   catalogue that the live board can never reward.
2. Upstream, the live board was monotone in sparsity while the catalogue proxy was
   anti-monotone (H19-5 121,131 dots -> 0.1922; d=1.5 60,069 -> 0.2477; d=2.8 44,090 -> 0.2600,
   owner reports; the proxy moved the other way).

So this repository fixes the budget by design at 40,000 dots with a 2.4 px minimum separation
and a hard exclusion of pixels within 200 m of the mapped catalogue, and labels it as a design
choice. It is *not* a holdout-optimal number and is not presented as one.
