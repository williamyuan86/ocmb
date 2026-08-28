# Experiments that directly interrogate T2′/T3′/T9′

The experiments below are designed as theorem-assumption/operating-curve tests, not as generic SOTA tables. They should be preregistered/frozen before reading the confirmatory seeds.

## Experiment 1 — Actual-selected-controls strict-LG FWER and power

### Question

Does the strict linear-Gaussian certification theorem remain calibrated when the held-out regression controls are the **actual PFCD-selected candidates** rather than oracle `Pa_i \ {j}` controls?

This directly tests the statistical obligations attached to T9′. The existing supplement validates Corollary C1 with oracle controls; this experiment closes that theorem-to-algorithm gap.

### Critical scope

Purely observational linear-Gaussian data do not identify a unique causal direction. Therefore the **confirmatory theorem lane supplies a valid topological predecessor order** (or an externally known temporal order) and tests the PFCD candidate/certification core end-to-end. An estimated-order lane may be reported as a stress diagnostic, but it must not be presented as confirming directed identifiability from linear-Gaussian observations.

### Algorithm under test

For every cross-fit training fold and target `i`:

1. Set `U_i` to the predecessors under the supplied valid topological order.
2. Build the protected candidate set
   `C_i = C_i^safe ∪ C_i^aux`, where the exact marginal theoretical view produces `C_i^safe`, auxiliary ridge/bootstrap views may add a frozen maximum `q_aux`, and no auxiliary ranking/cap can evict a safe candidate.
3. **Strict mode does not use group-Lasso as a veto.** Every candidate identity `j -> i` is eligible for held-out testing.
4. For tested edge `e=(j,i)`, use the actual selected control set
   `Q_e = C_i \ {j}`.
5. On the independent certification fold, run the ordinary partial-regression t test.
6. Assign `p=1` to every untested directed identity.
7. Merge folds with `p_e = min(1, K min_k p_{e,k})` and Bonferroni-test the fixed family `M*=d(d-1)` at family level `alpha=0.05`.

### Primary grid

Reuse the current strict-LG grid to make the new result directly comparable:

- `d in {20, 50}`;
- maximum indegree `s in {2, 4}`;
- certification-fold size `n_cert in {100, 250, 500, 1000}`;
- `beta_min in {0.1, 0.2, 0.3, 0.5}`;
- independent Gaussian SEM errors, sigma=1;
- randomized DAGs generated in a known topological order;
- 200 confirmatory seeds per setting initially, exactly matching the current 12,800-cell audit; if a Wilson interval is too wide to decide a gate, extend only that setting to 1000 seeds under a predeclared sequential rule.

Use the same coefficient-generation range as the current C1 validation so the only substantive change is **oracle controls -> actual PFCD-selected controls**.

### Two lanes

**E1-A: Confirmatory valid-order lane.** Supply the true/randomized topological order. This isolates T2′ + T9′ and makes `C_i subset ND_i` structural rather than estimated.

**E1-B: Estimated-order stress lane.** Run the production residual/BOSS front end, but report it only as a decomposition diagnostic. Measure order-admissibility failure separately; the theorem RHS becomes `alpha + delta_order + delta_cand` (or the directly measured joint `delta_good`). A large `delta_order` is a result, not something to hide.

### Quantities that must be logged per seed/fold/edge

- `G_order`: all supplied/estimated predecessor universes satisfy `Pa_i subset U_i subset ND_i`;
- `G_cand`: all parents are retained in `C_i`;
- `G_nd`: every selected candidate is a non-descendant;
- joint `G = G_order & G_cand & G_nd`;
- `|C_i|`, `|Q_e|`, rank/full-column-rank flag;
- residualized predictor norm `||z_e||_2`;
- realized degrees of freedom `nu_e`;
- realized noncentrality `lambda_e = |beta_e| ||z_e||_2 / sigma_i` for true edges;
- predicted noncentral-t power using the actual design;
- accepted/rejected identity and truth label.

### Primary estimands and theorem gates

1. **Conditional FWER on the good event**
   \[
   \widehat{FWER}_{G}=\Pr(\text{any false accepted}\mid G).
   \]
   Gate: no statistically significant evidence that it exceeds `0.05`; report Wilson or Clopper-Pearson interval.

2. **Unconditional T9′ bound**
   Estimate `delta_good = Pr(G^c)` and report
   \[
   \widehat{FWER} \quad\text{against}\quad 0.05+\widehat\delta_{good}.
   \]
   For a conservative empirical audit, compare the upper confidence bound of FWER to `0.05 +` the upper confidence bound for `delta_good`, and separately report the pointwise slack.

3. **Actual-design power calibration**
   For retained true parents, compare empirical acceptance probability to the exact noncentral-t prediction computed from realized `Q_e`, `nu_e`, and `||z_e||`. Report maximum absolute empirical-theoretical gap by setting and a reliability plot binned by predicted power.

4. **Selection cost of exactness**
   Report candidate recall, median/95th percentile `|C_i|`, certification tests, wall time, and the power loss relative to the existing oracle-control C1 experiment. This quantifies how much exact validity costs when nuisance controls are selected rather than known.

### What would falsify the proposed theorem interface?

- Conditional FWER exceeds 0.05 on cells where `G` and full-rank conditions hold: this attacks the strict-LG statistical derivation/implementation.
- A selected `Q_e` violates the control sandwich despite `Pa_i subset C_i subset ND_i`: this would contradict the deterministic T9′ lemma and should be impossible if logging is correct.
- Empirical power systematically disagrees with the actual-design noncentral-t curve on good/full-rank cells: this indicates a test implementation/specification error.
- High unconditional FWER that is explained entirely by a large `delta_good` does **not** falsify T9′; it falsifies the practical usefulness of the preceding order/candidate stage.

---

## Experiment 2 — Matched-recall hub complexity and residual discovery width

### Question

On a hub-induced correlation cloud where the original A7/B3 raw score-tail assumption fails, does correct/estimated hub peeling collapse discovery width and reduce candidate/certification work **at matched parent recall**?

This is the direct empirical test of T2′ + T3′ and replaces an unmatched workload comparison with an estimand that forces methods to recover the same local object.

### Family H0: existing theorem-valid control

Retain the current hub-valid family as a sanity/control lane. It verifies that residual peeling does not damage the regime where the raw discovery width is already small.

### Family H1: faithful Gaussian correlation-cloud hub

For `Delta` children, generate mutually independent

- `H ~ N(0,1)`;
- private parents `P_k ~ N(0,1)`;
- noises `epsilon_k ~ N(0, sigma^2)`;

and

\[
C_k=aH+bP_k+\epsilon_k.
\]

Primary parameters:

- `a=1`;
- `b=0.3`;
- `sigma=0.5`;
- `Delta in {8, 32, 64, 128, 256, 512}`;
- sample sizes `n in {500, 1000, 2000, 4000}`;
- 100 confirmatory seeds per cell after a separate calibration bank.

For standardized correlation scores,

\[
V=a^2+b^2+\sigma^2=1.34,
\]

\[
\rho(C_k,H)=a/\sqrt V\approx0.864,
\quad
\rho(C_k,P_k)=b/\sqrt V\approx0.259,
\quad
\rho(C_k,C_\ell)=a^2/V\approx0.746.
\]

Thus any raw threshold low enough to retain the private parent also admits every sibling at population level. For a child target the raw population discovery width is therefore `Delta+1` (hub, private parent, and `Delta-1` sibling distractors), i.e. `Theta(Delta)`.

After exact hub peeling,

\[
R_k=C_k-aH=bP_k+\epsilon_k,
\]

siblings have population residual correlation zero while

\[
|\rho(P_k,R_k)|=
\frac{b}{\sqrt{b^2+\sigma^2}}
\approx0.514.
\]

Hence the sibling contribution to residual discovery width is exactly zero, and under a positive threshold below 0.514 the population residual tail contains only the private parent in this family.

### Methods/variants

1. **PFCD raw:** current one-shot score/candidate path.
2. **PFCD-Peel (oracle anchor):** remove the true hub contribution first. This is the theorem-isolation lane for T3′.
3. **PFCD-Peel (estimated anchor):** select the first anchor by training-fold score/rank, estimate its linear coefficient on the training fold, residualize, then rescore. Log whether the selected anchor is the true hub. This is the deployable lane.
4. **IAMB** and **HITON-MB** using the same CI test family/sample.

For the matched-recall comparison, restrict the primary estimand to **child targets**. For a child with no descendants, its true Markov blanket is exactly its two parents `{H,P_k}`, so IAMB/HITON blanket recall and PFCD parent recall refer to the same truth set.

### Frozen matched-recall protocol

Do not tune each `Delta` on the confirmatory seeds.

1. Use a disjoint calibration bank spanning all declared `Delta,n` cells.
2. For each method, sweep only the predeclared threshold/budget grid and form a work-vs-recall curve.
3. Choose the **minimum-work frozen configuration** achieving pooled child-target recall at least `R*=0.95` on calibration data. Also report `R*=0.90` as sensitivity.
4. Freeze the configuration before confirmatory seeds.
5. On confirmatory data, report both achieved recall and work. If a method misses the target, mark the cell as **recall-infeasible** rather than comparing its cheaper work to methods that attained 0.95.

A complementary threshold-free surface should plot work required against recall over the full sweep; the frozen operating point is the confirmatory headline.

### Width measurements

For every child target record:

- analytic population raw width `omega_raw` (known exactly in H1);
- empirical raw width using an independent very-large reference draw or the closed-form population correlations;
- oracle-peel residual width `omega_resid_oracle`;
- estimated-peel residual width `omega_resid_est`;
- anchor correctness;
- protected safe-candidate size and auxiliary additions.

Primary T3′ prediction in H1:

\[
\omega_{raw}=\Delta+1=\Theta(\Delta),
\qquad
\omega_{resid,oracle}=1=O(1).
\]

The estimated-anchor lane should converge toward the oracle lane as anchor-selection/estimation error decreases with `n`; report the failure probability explicitly rather than folding it into width.

### Work metrics

Keep different computational primitives separate instead of inventing a single incomparable count:

- raw pair-score evaluations;
- retained candidate incidences per child target;
- held-out certification-test invocations per child target;
- IAMB/HITON CI-test invocations per child target;
- peak candidate memory;
- wall time as a secondary implementation metric.

The primary causal-complexity statement is about retained local width/certification work **conditional on matched recall**. Dense pair-score time remains a front-end cost and must be displayed separately.

### Confirmatory hypotheses

- **H2.1 Width law:** in H1, raw width slope versus `Delta` is approximately linear, while oracle-peel residual width is constant at the population level.
- **H2.2 Recall-preserving compression:** PFCD-Peel attains child-parent recall >=0.95 with candidate/certification counts whose growth in `Delta` is substantially below PFCD-raw and blanket search.
- **H2.3 Estimated-anchor bridge:** conditional on correct anchor selection, the estimated-peel width/work approaches the oracle-peel lane; unconditional degradation is explained by measured anchor failure.
- **H2.4 Honest runtime boundary:** if dense score scanning dominates wall time, report that explicitly. T3′ would still validate a local-width theorem, not an end-to-end subquadratic theorem.

### High-dimensional extension (secondary, not theorem-confirmatory)

After H1 succeeds, apply the same logging to `d in {1000,2000,4000,8000}` ER/SF scaling cells:

- one-shot LSH proposal;
- residual/peeling-aware proposal;
- candidate recall;
- discovery width before/after residualization;
- proposal recall, F1/AUPR, memory, time.

This tests whether the mechanism responsible for the hub-cloud result transfers to the current high-dimensional bottleneck. It is **not** direct evidence for T3′ unless the required residual structural assumptions are checked. If high-dimensional candidate recall does not improve, the paper should not claim that T3′ solved the LSH problem.

---

## Mapping theorem -> experiment

| Theorem component | Direct empirical quantity |
|---|---|
| T2′ parent inclusion | `Pa_i subset C_i^safe` frequency |
| T2′ candidate-tail inclusion | `C_i^safe subset N_i(tau_i)` on synthetic cells where population score is known |
| T2′ discovery-width bound | `|C_i^safe| <= omega_i(tau_i)` |
| protected auxiliary corollary | safe-parent retention after auxiliary union/cap; `|C_i| <= omega_i+q_aux` |
| T3′ sibling cancellation | residual sibling covariance/correlation after hub peel |
| T3′ private-parent preservation | residual parent covariance/correlation and retained-parent rate |
| T3′ scope/failure budget | anchor correctness and non-sibling residual distractors |
| T9′ control sandwich | `Pa_i\{j} subset Q_e subset ND_i\{j}` for every tested edge on `G` |
| T9′ exact-on-good-event FWER | empirical FWER conditional on `G` |
| T9′ unconditional composition | empirical FWER versus `alpha + delta_good` |
| T9′ power | empirical acceptance versus actual-design noncentral-t prediction |

The experiments therefore do not merely accompany the new theory: they expose exactly which assumptions make the theorems useful or vacuous in practice.
