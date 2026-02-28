# OSI Mechanism Investigation: Final Report

## 1. Executive Summary

This investigation systematically determined which mechanisms drive orientation
selectivity index (OSI) emergence in the RGC-LGN-V1 spiking network simulation.
Six mechanisms were tested via ablation, dose-response, and mechanistic pathway
analysis across 66+ conditions with 3 seeds each (198+ simulation runs).

**Key finding:** OSI emergence requires the combination of **Hebbian STDP** and
**heterosynaptic depression** — neither alone is sufficient. STDP alone produces
near-zero OSI (0.014), but adding heterosynaptic depression jumps it to 0.792.
The **ON/OFF split mechanism** provides significant additional enhancement when
ablated from the full model (delta = -0.324), though its marginal knock-in
contribution after heterosynaptic depression is small (+0.006). PV, SOM, and STP
are essentially neutral for OSI.

**Critical insight from knock-in vs ablation discrepancy:** Ablation of STDP
(from the full model) drops OSI to 0.200, not 0.014. This means the remaining
mechanisms (het, split, PV, SOM, STP) can produce weak orientation selectivity
even without STDP — likely via the split constraint's slow weight scaling.
Conversely, STDP alone (0.014) fails because without heterosynaptic depression,
all weights saturate at w_max, destroying selectivity. **Heterosynaptic
depression is the essential competitive mechanism that enables STDP to produce
oriented RFs.**

## 2. Mechanism Ranking (by ablation delta-OSI)

| Rank | Mechanism | Ablation OSI | Baseline (0.846) | Delta | Verdict |
|------|-----------|-------------|------------------|-------|---------|
| 1 | Triplet STDP (M1) | 0.200 | 0.846 | -0.646 | **ESSENTIAL** |
| 2 | ON/OFF Split (M3) | 0.522 | 0.846 | -0.324 | **IMPORTANT** (synergistic) |
| 3 | Heterosynaptic Depression (M2) | 0.745 | 0.846 | -0.101 | Facilitating |
| 4 | PV Inhibition (M4) | 0.824 | 0.846 | -0.022 | Modulatory |
| 5 | SOM Inhibition (M5) | 0.837 | 0.846 | -0.009 | Modulatory |
| 6 | TC STP (M6) | 0.844 | 0.846 | -0.002 | Neutral |

## 3. Detailed Mechanism Findings

### M1: Triplet STDP — ESSENTIAL
- Without any STDP, OSI = 0.200 (near chance). STDP is the core driver.
- **Pair STDP alone is sufficient** (OSI = 0.852). Triplet terms provide only ~2% boost.
- LTP/LTD ratio has an optimal range; too much LTD degrades selectivity.
- The multiplicative weight bounds (LTP proportional to w_max-W, LTD proportional to W) are critical
  for creating bimodal weight distributions that encode orientation.

### M2: Heterosynaptic Depression — FACILITATING
- Ablation reduces OSI by ~12% (0.846 -> 0.745). Important but not essential.
- Clear dose-response: OSI monotonically increases up to A_het=0.064 (OSI=0.911),
  then slightly declines at higher values.
- Mechanism: competitive weight depression of inactive synapses enhances winner/loser
  separation, sharpening receptive fields.
- Optimal A_het=0.064 (~2x default) provides best selectivity.

### M3: ON/OFF Split — IMPORTANT (synergistic)
- Ablation reduces OSI by ~38% (0.846 -> 0.522). Second most impactful mechanism.
- **Critical finding: STDP split alone is DESTRUCTIVE** (OSI=0.062).
- **Constraint alone is weak** (OSI=0.352).
- **Together they are super-additive**: the synergy between STDP-driven ON/OFF
  decorrelation and the stabilizing constraint produces the large OSI boost.
- Inverted-U dose-response: peaks at A_split~0.4, default 0.2 is near-optimal.

### M4: PV Inhibition — MODULATORY
- Ablation reduces OSI by only 2.6% (0.846 -> 0.824). Not essential.
- Mild inverted-U: optimal at w_pv_e=0.25, slight degradation at w_pv_e=4.0.
- iSTDP (plastic PV) vs fixed PV: negligible difference for OSI.
- PV's biological role (gain control) doesn't meaningfully interact with
  orientation selectivity formation in this model.

### M5: SOM Lateral Inhibition — MODULATORY
- Ablation reduces OSI by only 1.1% (0.846 -> 0.837). Not essential.
- Orientation diversity (resultant R) is already low (~0.1) without SOM,
  suggesting the golden-ratio training schedule provides sufficient diversity.
- SOM's biological role (cross-ensemble competition) is largely redundant with
  the diverse stimulus schedule at M=16.

### M6: TC STP — NEUTRAL
- Ablation has zero effect (0.844 vs 0.846). STP is irrelevant for OSI.
- Dose-response flat from u=0 to u=0.2. Only catastrophic at u=0.5 (OSI=0.0),
  where thalamocortical drive is essentially eliminated.
- STP's biological role (temporal gain normalization) does not interact with
  the orientation selectivity pathway.

## 4. Additive Knock-In Results

Starting from STDP-only (all other mechanisms off), mechanisms added one at a time:

| Step | Configuration | OSI | Marginal delta |
|------|--------------|-----|----------------|
| 1 | KI_1_stdp_only | 0.014 |  |
| 2 | KI_2_plus_het | 0.792 | +0.779 |
| 3 | KI_3_plus_split | 0.798 | +0.006 |
| 4 | KI_4_plus_pv | 0.829 | +0.031 |
| 5 | KI_5_plus_som | 0.844 | +0.015 |
| 6 | KI_6_full_model | 0.846 | +0.002 |

**Critical observation:** STDP alone produces near-zero OSI (0.014). This
contradicts the ablation result where removing STDP (from full model) still
leaves OSI = 0.200. The discrepancy reveals that:

1. **STDP alone is insufficient** — without heterosynaptic depression to
   prevent weight saturation, all synapses grow to w_max and orientation
   selectivity cannot emerge.
2. **Heterosynaptic depression is the key enabler** — adding it to STDP
   produces the largest single jump (+0.779), accounting for ~95% of
   final OSI.
3. **The remaining mechanisms (split, PV, SOM, STP) collectively add only
   +0.054** — a ~6% marginal improvement over STDP+Het alone.
4. **Ablation vs knock-in asymmetry for Split**: Ablating split from
   the full model drops OSI by 0.324 (large), but adding split after
   Het only adds 0.006 (tiny). This indicates **strong redundancy between
   Het and Split** — both are competitive weight mechanisms, and either
   alone is nearly sufficient when paired with STDP.

## 5. Pairwise Interaction Analysis

I(A,B) = OSI(both) - OSI(A_only) - OSI(B_only) + OSI(neither)
Positive = synergistic, Negative = redundant, ~0 = additive

| Pair | I(A,B) | Interpretation |
|------|--------|----------------|
| HetxSplit | -0.653 | **Strongly redundant** |
| HetxPV | -0.208 | Redundant |
| SplitxPV | +0.042 | Mildly synergistic |

The strong redundancy between Het and Split (I = -0.653) confirms that both
serve as competitive weight mechanisms. When both are present, much of
Split's contribution is already covered by Het (and vice versa). This
explains the large ablation-vs-knockin asymmetry for Split noted above.

## 6. Causal Pathway Diagram

```
Stimulus (drifting grating)
    |
    v
RGC (ON/OFF center-surround) -> LGN (Izhikevich relay)
    |
    v
LGN -> V1 thalamocortical synapses (W)
    |
    |--- [ESSENTIAL] Pair STDP: co-active inputs strengthen (LTP),
    |    anti-correlated inputs weaken (LTD). Multiplicative bounds
    |    create bimodal weight distributions -> oriented RFs.
    |
    |--- [IMPORTANT] ON/OFF Split (STDP + constraint synergy):
    |    STDP split decorrelates ON/OFF subfields, constraint prevents
    |    collapse. Together: push-pull RF structure -> enhanced grating
    |    selectivity.
    |
    |--- [FACILITATING] Heterosynaptic depression:
    |    Post-spike depression of inactive synapses enhances competitive
    |    separation of winners vs losers -> sharper selectivity.
    |
    |--- [MODULATORY] PV inhibition: mild gain control, not essential.
    |--- [MODULATORY] SOM inhibition: inter-ensemble competition,
    |    redundant with diverse stimulus schedule.
    |--- [NEUTRAL] TC STP: temporal gain normalization, orthogonal to OSI.
    |
    v
Oriented weight structure (W) -> Orientation Selectivity (OSI)
```

## 7. Dose-Response Key Thresholds

| Parameter | Optimal | Default | Effect of optimal |
|-----------|---------|---------|-------------------|
| A2_minus (LTD) | ~0.010 | 0.010 | Default is near-optimal |
| A_het | 0.064 | 0.032 | +7.7% OSI (0.846->0.911) |
| A_split | 0.2-0.4 | 0.2 | Default is near-optimal |
| w_pv_e | 0.25 | 1.0 | Negligible effect |
| w_som_e | any | 0.05 | Negligible effect |
| tc_stp_u | 0-0.1 | 0.05 | No effect (flat) |

## 8. Methodological Notes

- **Model**: M=16 ensembles, N=8 patch size, 300 training segments
- **Seeds**: 3 per condition (1, 42, 137)
- **Training**: Golden-ratio orientation schedule (low-discrepancy)
- **Evaluation**: 12 orientations, 3 repeats per orientation
- **Baseline OSI**: 0.846 +/- 0.008
- **Total runs**: ~216+ simulation runs across all conditions
- **Compute**: ~25 hours wall-clock (6 parallel agents on 28 cores)
