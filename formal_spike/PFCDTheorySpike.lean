import Mathlib

open Set MeasureTheory

namespace PFCDTheorySpike

noncomputable section

/-!
# T2′: discovery width

This is the deterministic score-gap core.  It deliberately does *not* assume
that the population tail is O(s log d).  The tail cardinality is exposed as
the discovery width.
-/
section T2Prime

variable {ι : Type*} [DecidableEq ι]

def safeCandidates
    (U : Finset ι) (scoreHat : ι → ℝ) (tauHat eps : ℝ) : Finset ι :=
  U.filter fun j => tauHat + 2 * eps ≤ scoreHat j

def populationTail
    (U : Finset ι) (score : ι → ℝ) (tau : ℝ) : Finset ι :=
  U.filter fun j => tau ≤ score j

/--
T2′ (Discovery-width sure screening).
Uniform score/boundary error plus a 4 eps parent margin implies:
(i) every parent enters the safe candidate set;
(ii) every safe candidate lies in the population tail at tau; hence
(iii) candidate cardinality is at most the discovery width.
-/
theorem t2prime_discovery_width
    (U P : Finset ι)
    (score scoreHat : ι → ℝ)
    (tau tauHat eps : ℝ)
    (hPsub : P ⊆ U)
    (hscore : ∀ j ∈ U, |scoreHat j - score j| ≤ eps)
    (hboundary : |tauHat - tau| ≤ eps)
    (hdetect : ∀ p ∈ P, tau + 4 * eps ≤ score p) :
    P ⊆ safeCandidates U scoreHat tauHat eps ∧
    safeCandidates U scoreHat tauHat eps ⊆ populationTail U score tau ∧
    (safeCandidates U scoreHat tauHat eps).card ≤
      (populationTail U score tau).card := by
  have hParents : P ⊆ safeCandidates U scoreHat tauHat eps := by
    intro p hp
    have hpU : p ∈ U := hPsub hp
    have hs := abs_le.mp (hscore p hpU)
    have hb := abs_le.mp hboundary
    have hd := hdetect p hp
    apply Finset.mem_filter.mpr
    refine ⟨hpU, ?_⟩
    linarith
  have hSafeTail :
      safeCandidates U scoreHat tauHat eps ⊆ populationTail U score tau := by
    intro j hj
    have hj' : j ∈ U ∧ tauHat + 2 * eps ≤ scoreHat j := by
      simpa [safeCandidates] using (Finset.mem_filter.mp hj)
    have hs := abs_le.mp (hscore j hj'.1)
    have hb := abs_le.mp hboundary
    apply Finset.mem_filter.mpr
    refine ⟨hj'.1, ?_⟩
    linarith
  exact ⟨hParents, hSafeTail, Finset.card_le_card hSafeTail⟩

/--
Protected multi-view composition: auxiliary views may add at most qAux
candidates, but can never evict a safe candidate.  Therefore parent recall is
preserved and the final size is bounded by discovery width + qAux.
-/
theorem t2prime_protected_aux
    (P safe aux : Finset ι) (width qAux : ℕ)
    (hParents : P ⊆ safe)
    (hSafeWidth : safe.card ≤ width)
    (hAux : aux.card ≤ qAux) :
    P ⊆ safe ∪ aux ∧ (safe ∪ aux).card ≤ width + qAux := by
  constructor
  · intro p hp
    exact Finset.mem_union_left aux (hParents hp)
  · calc
      (safe ∪ aux).card ≤ safe.card + aux.card := Finset.card_union_le safe aux
      _ ≤ width + qAux := Nat.add_le_add hSafeWidth hAux

end T2Prime

/-!
# T3′: hub residual peeling

The Lean layer proves the covariance algebra, not distribution-specific
independence.  Independence/uncorrelatedness is represented by explicit zero
cross-covariance hypotheses, matching the paper's existing formalization
policy of keeping statistical obligations visible.
-/
section T3Prime

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

structure CovLike (V : Type*) [AddCommGroup V] [Module ℝ V] where
  cov : V → V → ℝ
  add_left : ∀ x y z, cov (x + y) z = cov x z + cov y z
  add_right : ∀ x y z, cov x (y + z) = cov x y + cov x z
  smul_left : ∀ a x y, cov (a • x) y = a * cov x y
  smul_right : ∀ a x y, cov x (a • y) = a * cov x y

namespace CovLike

lemma two_term
    (K : CovLike V) (a b : ℝ) (x y z w : V) :
    K.cov (a • x + y) (b • z + w) =
      a * b * K.cov x z + a * K.cov x w + b * K.cov y z + K.cov y w := by
  simp only [K.add_left, K.add_right, K.smul_left, K.smul_right]
  ring

end CovLike

def child (a b : ℝ) (H P E : V) : V :=
  a • H + b • P + E

def peelHub (a : ℝ) (H C : V) : V :=
  C - a • H

lemma peel_child (a b : ℝ) (H P E : V) :
    peelHub a H (child a b H P E) = b • P + E := by
  unfold peelHub child
  abel

/--
T3′, pairwise algebraic core: after subtracting the correctly identified hub
component from two children, their covariance is zero when their private
parents/noises have zero cross-covariances.
-/
theorem t3prime_hub_sibling_cancel
    (K : CovLike V)
    (ak bk al bl : ℝ)
    (H Pk Pl Ek El : V)
    (hPP : K.cov Pk Pl = 0)
    (hPE : K.cov Pk El = 0)
    (hEP : K.cov Ek Pl = 0)
    (hEE : K.cov Ek El = 0) :
    K.cov
        (peelHub ak H (child ak bk H Pk Ek))
        (peelHub al H (child al bl H Pl El)) = 0 := by
  rw [peel_child, peel_child, K.two_term]
  rw [hPP, hPE, hEP, hEE]
  ring

/--
T3′, signal preservation: hub peeling leaves the private-parent covariance
bk * Var(Pk), provided the private parent is uncorrelated with the child noise.
-/
theorem t3prime_private_parent_preserved
    (K : CovLike V)
    (ak bk : ℝ)
    (H Pk Ek : V)
    (hPE : K.cov Pk Ek = 0) :
    K.cov Pk (peelHub ak H (child ak bk H Pk Ek)) =
      bk * K.cov Pk Pk := by
  rw [peel_child, K.add_right, K.smul_right, hPE]
  ring

/-- A zero residual score cannot survive a strictly positive threshold. -/
theorem t3prime_zero_sibling_below_positive_threshold
    (tau : ℝ) (htau : 0 < tau) : ¬ tau ≤ |(0 : ℝ)| := by
  simpa using htau

end T3Prime

/-!
# T9′: actual-selected-control strict-LG composition

Two pieces are machine checked here:
1. if the selected candidate set contains all parents and only
   non-descendants, then using C \ {j} as the held-out regression control set
   automatically satisfies the strict-LG control-set sandwich;
2. exact-on-good-event FWER/power statements compose unconditionally by
   charging the candidate/order failure event.

The Gaussian t distribution itself remains a statistical ingredient, exactly
as Corollary C1 is outside the current Lean coverage in the manuscript.
-/
section T9Prime

variable {ι : Type*} [DecidableEq ι]

/-- Actual PFCD-selected controls satisfy the C1 sandwich on a good candidate event. -/
theorem t9prime_selected_controls_valid
    (P C ND : Finset ι) (j : ι)
    (hPC : P ⊆ C) (hCND : C ⊆ ND) :
    P.erase j ⊆ C.erase j ∧ C.erase j ⊆ ND.erase j := by
  constructor
  · intro x hx
    simp only [Finset.mem_erase] at hx ⊢
    exact ⟨hx.1, hPC hx.2⟩
  · intro x hx
    simp only [Finset.mem_erase] at hx ⊢
    exact ⟨hx.1, hCND hx.2⟩

variable {Ω : Type*} [MeasurableSpace Ω]

/--
If FWER on the good candidate/order event has joint probability at most alpha,
and the good event fails with probability at most deltaGood, then unconditional
FWER is at most alpha + deltaGood (written in ENNReal form).
-/
theorem t9prime_fwer_event_bound
    (μ : Measure Ω) (F G : Set Ω) (alpha deltaGood : ℝ)
    (hBad : μ Gᶜ ≤ ENNReal.ofReal deltaGood)
    (hFalseGood : μ (F ∩ G) ≤ ENNReal.ofReal alpha) :
    μ F ≤ ENNReal.ofReal deltaGood + ENNReal.ofReal alpha := by
  have hsub : F ⊆ Gᶜ ∪ (F ∩ G) := by
    intro ω hF
    by_cases hG : ω ∈ G
    · exact Or.inr ⟨hF, hG⟩
    · exact Or.inl hG
  calc
    μ F ≤ μ (Gᶜ ∪ (F ∩ G)) := measure_mono hsub
    _ ≤ μ Gᶜ + μ (F ∩ G) := measure_union_le _ _
    _ ≤ ENNReal.ofReal deltaGood + ENNReal.ofReal alpha :=
      add_le_add hBad hFalseGood

/--
Recovery-failure composition.  On the good event, failure is covered by a
false-positive event F or a true-edge miss event M.  The theorem charges the
good-event failure probability separately.
-/
theorem t9prime_recovery_event_bound
    (μ : Measure Ω) (R F M G : Set Ω)
    (alpha beta deltaGood : ℝ)
    (hCover : R ⊆ Gᶜ ∪ (F ∩ G) ∪ (M ∩ G))
    (hBad : μ Gᶜ ≤ ENNReal.ofReal deltaGood)
    (hFalseGood : μ (F ∩ G) ≤ ENNReal.ofReal alpha)
    (hMissGood : μ (M ∩ G) ≤ ENNReal.ofReal beta) :
    μ R ≤ ENNReal.ofReal deltaGood + ENNReal.ofReal alpha + ENNReal.ofReal beta := by
  calc
    μ R ≤ μ (Gᶜ ∪ (F ∩ G) ∪ (M ∩ G)) := measure_mono hCover
    _ ≤ μ (Gᶜ ∪ (F ∩ G)) + μ (M ∩ G) := measure_union_le _ _
    _ ≤ (μ Gᶜ + μ (F ∩ G)) + μ (M ∩ G) := by
      gcongr
      exact measure_union_le _ _
    _ ≤ (ENNReal.ofReal deltaGood + ENNReal.ofReal alpha) + ENNReal.ofReal beta :=
      add_le_add (add_le_add hBad hFalseGood) hMissGood

end T9Prime

#print axioms t2prime_discovery_width
#print axioms t2prime_protected_aux
#print axioms t3prime_hub_sibling_cancel
#print axioms t3prime_private_parent_preserved
#print axioms t9prime_selected_controls_valid
#print axioms t9prime_fwer_event_bound
#print axioms t9prime_recovery_event_bound

end PFCDTheorySpike
