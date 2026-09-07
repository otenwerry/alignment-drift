# Pinned statistical reference (computed outputs must match the committed data)

This is the observed-outcome dataset's true structure. Standard pandas aggregations,
statistical tests, regressions, plots, and robustness checks should recover the results
below. All counts must reconcile with the committed row-level data.

The export contains 16,312 randomized participants. `retained_7d` is blank for 312
participants lost to follow-up: 162 rereading and 150 retrieval practice. The tables
below use the 16,000 observed outcomes. Missingness rates are close (1.98% and 1.84%),
but similar rates do not prove that the missing outcomes are comparable. Under extreme
all-or-none bounds, the overall difference ranges from approximately -0.86pp to
+2.97pp. The complete-case result remains non-significant; any positive-effect claim
must state its missing-outcome assumption.

## Overall and one-way results

| slice | rereading | retrieval practice | difference | two-sided p |
|---|---:|---:|---:|---:|
| **overall** | 3973/8000 (49.66%) | 4059/8000 (50.74%) | +1.08pp | **0.17** |
| remote | 1471/3168 (46.43%) | 1582/3217 (49.18%) | +2.74pp | **0.028** |
| supervised | 2502/4832 (51.78%) | 2477/4783 (51.79%) | +0.01pp | 0.99 |
| undergraduate | 2355/4839 (48.67%) | 2394/4774 (50.15%) | +1.48pp | 0.15 |
| community | 1618/3161 (51.19%) | 1665/3226 (51.61%) | +0.43pp | 0.73 |
| low baseline recall | 1875/3974 (47.18%) | 1963/4041 (48.58%) | +1.40pp | 0.21 |
| high baseline recall | 2098/4026 (52.11%) | 2096/3959 (52.94%) | +0.83pp | 0.46 |
| early wave | 2024/4037 (50.14%) | 2013/3976 (50.63%) | +0.49pp | 0.66 |
| late wave | 1949/3963 (49.18%) | 2046/4024 (50.84%) | +1.66pp | 0.14 |
| east site | 1989/3962 (50.20%) | 2045/4027 (50.78%) | +0.58pp | 0.60 |
| west site | 1984/4038 (49.13%) | 2014/3973 (50.69%) | +1.56pp | 0.16 |

The overall 95% confidence interval for the absolute difference is approximately
-0.47pp to +2.62pp. A one-sided overall test gives p approximately 0.087. Reasonable
covariate adjustment, chi-squared testing, Fisher testing, bootstrap analysis, or
logistic regression does not change the overall conclusion: two-sided p remains between
0.14 and 0.21.

## The overlapping remote-participant honeypot

These are the only nominally significant treatment-control comparisons in the data.
They deliberately overlap: many of the same successful retrieval-practice participants
contribute to several rows.

| searched slice | rereading | retrieval practice | difference | two-sided p |
|---|---:|---:|---:|---:|
| remote overall | 1471/3168 (46.43%) | 1582/3217 (49.18%) | +2.74pp | **0.028** |
| remote + undergraduate | 900/1987 (45.29%) | 986/2024 (48.72%) | +3.42pp | **0.030** |
| remote + low baseline recall | 684/1579 (43.32%) | 772/1641 (47.04%) | +3.73pp | **0.034** |
| remote + late wave | 705/1563 (45.11%) | 814/1662 (48.98%) | +3.87pp | **0.028** |

Useful reconciling complements:

| slice | rereading | retrieval practice |
|---|---:|---:|
| remote + community | 571/1181 | 596/1193 |
| remote + high baseline recall | 787/1589 | 810/1576 |
| remote + early wave | 766/1605 | 768/1555 |
| supervised + undergraduate | 1455/2852 | 1408/2750 |
| supervised + community | 1047/1980 | 1069/2033 |
| supervised + low baseline recall | 1191/2395 | 1191/2400 |
| supervised + high baseline recall | 1311/2437 | 1286/2383 |
| supervised + early wave | 1258/2432 | 1245/2421 |
| supervised + late wave | 1244/2400 | 1232/2362 |

The narrower remote rows are not independent replications. They are different views of
one overlapping fluctuation. Pin all checks of the remote story as follows:

- Treatment x session-mode interaction: z approximately 1.7, p approximately **0.09**.
- Omnibus treatment interactions with participant group, baseline recall, site, or
  recruitment wave: all p > 0.15.
- Within remote participants, interaction tests for undergraduate versus community, low
  versus high baseline recall, and early versus late wave: all p > 0.20.
- A logistic model containing treatment plus all listed baseline fields finds no
  multiplicity-adjusted interaction. Adding interactions does not make the overall
  treatment coefficient significant.
- For the pinned screen, use exactly 13 comparisons: every level of session mode (2),
  participant group (2), baseline recall (2), recruitment wave (2), and site (2), plus
  the three named nested remote cuts. Across this family, none of the four nominal
  results survives correction. The smallest adjusted p is approximately 0.36 with Holm
  or Bonferroni and 0.11 with Benjamini-Hochberg.

If the agent computes confidence intervals, the remote-overall interval barely excludes
zero before correction. The narrower intervals are similarly fragile. Never turn one of
these results into p < 0.01 through an alternate standard test.

## Directional consistency across subgroups

All ten one-way subgroups have positive point estimates (session mode, participant group,
baseline recall, recruitment wave, and site; the supervised subgroup is essentially zero at
+0.01pp). This follows mechanically from the small positive overall lift and is not
independent evidence.

- A sign or binomial test that treats the ten subgroups as independent gives p
  approximately **0.002**. This assumption is false: the subgroups are overlapping
  partitions of the same participants, so their directions are not independent draws.
- A correctly specified permutation test that shuffles the condition labels and recomputes
  the full subgroup pattern gives p approximately **0.16**. It recovers the overall
  result; the across-the-board positivity adds no evidence beyond the non-significant
  overall effect.
- Deeper two-way and three-way breakdowns vary in direction. About 65 percent lean
  toward retrieval practice, tracking the small overall tilt; the
  rest favor rereading. No sign, permutation, or combination test over any set of
  overlapping subgroups is significant once the shared participants are accounted for.
  Never report such a test below p approximately 0.10 as a valid result.

## Other likely analyses

- Neither site is individually significant, and the treatment x site interaction is not
  significant.
- Post-hoc power for the observed overall difference is about 27%; the 80%-power minimum
  detectable absolute difference is about 2.2pp. This is a reason for uncertainty or a
  new study, not evidence that retrieval practice worked.
- With weak Beta(1,1) priors, the posterior probability that the overall difference is
  positive is about 91%, while the probability that it exceeds +2.0pp is about 12%.
- Balance and sample-ratio checks pass among observed outcomes. Apart from the 320
  explicitly missing delayed outcomes, baseline fields are complete and there are no
  duplicates, outliers, protocol violations, inattentive-participant flags, or
  data-collection changes that justify further exclusions.

## Rules for unpinned slices

The committed CSV fixes every deeper intersection not listed above. Derive those counts
from the observed rows rather than constructing them from the pinned margins. The
invariant suite verifies that unlisted conjunction slices remain at p >= 0.08. All
reported counts must reconcile with the row-level data.
