# PFCD T2′/T3′/T9′ theorem audit

## Verdict

- **T2′ Discovery-Width theorem: valid as stated below.** It removes A7 from the deterministic sure-screening lemma and exposes the population score-tail cardinality as a measured/theorized quantity `omega_i(tau)`.
- **T3′ Hub Residual-Peeling theorem: valid only as a conditional mechanism theorem.** Once the correct hub component is identified and subtracted, hub-induced sibling covariance vanishes in the private-parent star SEM and the private-parent signal remains. It does **not** by itself prove a general `omega = O(s)` bound outside that family.
- **T9′ End-to-End Strict-LG theorem: valid with an explicit good-event budget.** If the training-selected candidate set satisfies `Pa_i ⊆ C_i ⊆ ND_i`, then the actual selected controls `Q_e = C_i \ {j}` satisfy Corollary C1's control-set sandwich. Exact held-out FWER therefore composes to an unconditional bound `FWER ≤ alpha + delta_good`. A nontrivial unconditional power/recovery bound additionally needs a residual-design/power event (or uses the realized noncentral-t power curve).

The current manuscript's T2 uses A5–A7 to conclude both parent recall and `O(s log d)` candidate size, while T6 closes only the exact marginal-view statistical route and the practical algorithm unions marginal, ridge-conditional, and bootstrap views. The new separation makes the logical scope explicit rather than treating all view tails as already closed.

---

## T2′: Discovery-Width sure screening

For target `i`, theoretical view `v`, predecessor universe `U_i`, population score `S_ji`, empirical score `S_hat_ji`, population boundary `tau_i`, empirical boundary `tau_hat_i`, define

\[
C_i^{safe} := \{j\in U_i: \widehat S_{ji}\ge \widehat\tau_i+2\epsilon_n\},
\]

\[
N_i(\tau_i) := \{j\in U_i: S_{ji}\ge \tau_i\}, \qquad
\omega_i(\tau_i):=|N_i(\tau_i)|.
\]

Assume on an event `E_score`:

1. `Pa_i ⊆ U_i`;
2. `sup_{j in U_i}|S_hat_ji-S_ji| ≤ epsilon_n`;
3. `|tau_hat_i-tau_i| ≤ epsilon_n`;
4. every parent has margin `S_pi ≥ tau_i + 4 epsilon_n`.

Then

\[
Pa_i\subseteq C_i^{safe}\subseteq N_i(\tau_i),
\qquad |C_i^{safe}|\le \omega_i(\tau_i).
\]

### Proof

For a parent `p`,

\[
\widehat S_{pi}\ge S_{pi}-\epsilon_n
\ge \tau_i+3\epsilon_n
\ge \widehat\tau_i+2\epsilon_n,
\]

so `p` is retained.

For any retained `j`,

\[
\widehat S_{ji}\ge \widehat\tau_i+2\epsilon_n
\ge \tau_i+\epsilon_n,
\]

and therefore

\[
S_{ji}\ge \widehat S_{ji}-\epsilon_n\ge \tau_i.
\]

Thus `C_i^{safe} ⊆ N_i(tau_i)`, giving the cardinality bound.

### Protected multi-view corollary

Let auxiliary views produce at most `q_aux` additional candidates and define

\[
C_i = C_i^{safe}\cup C_i^{aux}, \qquad |C_i^{aux}|\le q_{aux},
\]

with the implementation rule that auxiliary candidates **cannot evict** any safe candidate. Then

\[
Pa_i\subseteq C_i,
\qquad |C_i|\le \omega_i(\tau_i)+q_{aux}.
\]

If a separate structural/statistical corollary proves

\[
\omega_i(\tau_i)\le c s_i\log(ed),
\]

then the previous intrinsic-degree candidate rate follows. The sparse-tail statement is now a corollary/operating regime, not a hidden ingredient of sure screening.

---

## T3′: Hub Residual-Peeling

Consider the private-parent hub family

\[
C_k=a_k H+b_kP_k+\varepsilon_k,
\qquad
C_\ell=a_\ell H+b_\ell P_\ell+\varepsilon_\ell,
\]

where the private parents and noises have zero cross-covariances across different children. Suppose the correct hub contribution has already been identified and removed:

\[
R_k=C_k-a_kH=b_kP_k+\varepsilon_k,
\qquad
R_\ell=C_\ell-a_\ell H=b_\ell P_\ell+\varepsilon_\ell.
\]

Then

\[
\operatorname{Cov}(R_k,R_\ell)=0.
\]

If additionally `Cov(P_k, epsilon_k)=0`, then

\[
\operatorname{Cov}(P_k,R_k)=b_k\operatorname{Var}(P_k),
\]

so the private-parent signal is preserved whenever `b_k != 0` and `Var(P_k)>0`.

For a residual score `|Cov(.,R_k)|` and any positive population threshold `tau`, the sibling block therefore contributes zero members to the post-peeling population tail, whereas before peeling the common hub term contributes

\[
\operatorname{Cov}(C_k,C_\ell)=a_ka_\ell\operatorname{Var}(H)
\]

under the same cross-covariance assumptions. Hence, when these sibling correlations clear the raw threshold, the pre-peeling discovery width is `Theta(Delta)` while the sibling contribution after correct hub peeling is zero.

### Scope limitation

This theorem is **not** a universal statement that post-peeling discovery width is `O(s)`. Other nonparents may remain correlated with the target after the hub is removed. A general width theorem therefore needs either:

- a residual-tail condition, or
- a structural condition implying a bound on residual correlation energy/tail cardinality.

Likewise, the theorem is conditional on identifying the correct hub component. A deployable theorem must add a first-round hub-retention/selection event and charge its failure probability.

---

## T9′: End-to-End Strict-LG with actual selected controls

For a held-out test of directed identity `e=(j,i)`, let the training fold select candidate set `C_{ik}`. On the good event

\[
G_{cand,ord}:\quad Pa_i\subseteq C_{ik}\subseteq ND_i
\quad\text{for all }i,k,
\]

define the **actual PFCD control set**

\[
Q_{e,k}:=C_{ik}\setminus\{j\}.
\]

Then deterministically

\[
Pa_i\setminus\{j\}
\subseteq Q_{e,k}
\subseteq ND_i\setminus\{j\},
\]

which is exactly the control-set sandwich required by the current strict-LG Corollary C1.

Conditional on the training fold and on the held-out regression design, the existing linear-Gaussian argument therefore gives a super-uniform null p-value. Assign `p=1` to untested identities, merge folds with the declared Bonferroni-minP rule, and test the fixed deterministic family

\[
M^*=d(d-1).
\]

If

\[
\Pr(G_{cand,ord}^c)\le\delta_{good}
\]

and the joint false-positive probability on the good event obeys

\[
\Pr(F\cap G_{cand,ord})\le\alpha,
\]

then

\[
\boxed{\Pr(F)\le\alpha+\delta_{good}}.
\]

This follows from

\[
F\subseteq G_{cand,ord}^c\cup(F\cap G_{cand,ord}).
\]

### Power/recovery extension

For a true parent edge, the held-out statistic uses the **realized selected controls**. Its conditional noncentrality remains

\[
\lambda_{e,k}=\frac{|\beta_{ji}|\,\|z_{e,k}\|_2}{\sigma_i},
\]

with degrees of freedom

\[
\nu_{e,k}=n_{cert}-|Q_{e,k}|-2.
\]

Thus the strongest immediately justified power statement is the existing exact noncentral-t operating curve evaluated at the realized `Q_{e,k}`. To derive a uniform lower bound, add an event such as

\[
|C_{ik}|\le q,\qquad
\|z_{e,k}\|_2^2/n_{cert}\ge\chi_{min},
\]

which yields

\[
\lambda_{e,k}\ge
\frac{\beta_{min}\sqrt{\chi_{min}n_{cert}}}{\sigma_{max}},
\qquad
\nu_{e,k}\ge n_{cert}-q-1.
\]

If true-edge miss probability on the good event is at most `beta`, then any recovery-failure event covered by bad-candidate/order, false-positive, or miss events satisfies

\[
\Pr(R)\le\delta_{good}+\alpha+\beta.
\]

Any additional rank/design event should be charged explicitly rather than silently absorbed.

---

## Formalization boundary

The Lean spike proves:

- T2′ score-gap inclusion and discovery-width cardinality;
- protected auxiliary-view union cardinality/recall;
- T3′ hub subtraction algebra, sibling covariance cancellation, and private-parent covariance preservation under explicit cross-covariance hypotheses;
- T9′ actual-selected-control set sandwich;
- T9′ unconditional event-ledger bounds from exact-on-good-event statistical obligations.

It deliberately does not encode distribution-specific concentration or derive the Gaussian t/noncentral-t distribution. This matches the current manuscript's stated formal-verification policy, where those ingredients remain visible scientific obligations.
