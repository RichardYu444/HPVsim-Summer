#!/usr/bin/env python3
"""
demo_age_community_distributions
================================

Drive the **age + community** bipartite network
(``age_community_bipartite_network_model``) and plot its key distributions.
As in ``demo_bipartite_distributions.py`` the parameters are first
**calibrated** and the network **burned in** before measurement, so everything
is measured at the stationary regime. Demography is left **off** here (closed
population) to keep the stationary distributions clean; see the commented block
near ``simulate`` for how to inject external births/deaths.

What is new relative to the simple-toy demo
-------------------------------------------
* **Degree distribution on three timescales** (partners counted over three
  windows), the canonical sexual-network view:
    - *instantaneous* : partners held at a single instant (monthly momentary degree);
    - *annual*        : distinct partners over 12 months;
    - *lifetime*      : distinct partners over the whole measurement window.
* **Degree distribution split by relationship TYPE** (short / long / total),
  instantaneous.
* **Age-mixing bands**: the input age-preference kernel A and the *realised*
  partner-age mixing (row-normalised band x band matrix, i.e. the conditional
  P(partner band | own band)).
* **Community mixing**: the realised community x community mixing matrix.

Panels (2 x 4)
--------------
1. Degree distribution: instantaneous vs annual vs lifetime.
2. Instantaneous degree split by type: short / long / total.
3. Input age preference A (band_man x band_woman).
4. Realised partner-age mixing (row-normalised) + recovery check.
5. Realised community mixing (row-normalised).
6. Realised partnership durations, split by type (short vs long).
7. Mean distinct-partners-per-year over time vs the target input.
8. Standing fraction of long ties over time vs the input target.

Run:  python demo_age_community_distributions.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import age_community_bipartite_network_model as M

# =====================================================================
# PARAMETERS  (everything you might want to tweak lives here)
# =====================================================================
N_MEN   = 5000          # number of men   (U)
N_WOMEN = 5000          # number of women (V)  -> balanced heterosexual pop.
YEARS   = 35            # years of MEASUREMENT (after burn-in); sets the
                        # "lifetime" window (closed population here)
SEED = 0

USER_PARAMS = {
    "mean_partners_per_year": 2,    # target distinct partners / person / year
    "gamma_shape":            1.0,    # Gamma shape of degree dist (CV ~ 1/sqrt)
    "D_mean_short":           3.0,    # casual tie mean duration (months)
    "D_mean_long":            180,   # steady tie mean duration (months = 3 yr)
    "pi_long":                0.1,    # target STANDING fraction of long ties
    # --- age + community structure ---
    "n_communities":          1,
    "age_sigma":              7.0,    # age-kernel spread (years)
    "age_male_older":         3.0,    # men prefer women ~3 yr younger
    "community_off_diag":     0.1,    # off-diagonal community weight
    # default age bands (16-25,25-35,...,65-74)
}

OUT_PNG = "age_community_distributions.png"
_KEY = np.int64(1) << 33
SHORT, LONG = M.EDGE_SHORT, M.EDGE_LONG

# =====================================================================
# helpers
# =====================================================================

def inst_degrees_by_type(state):
    """Instantaneous per-person degree, split into short / long (and total)."""
    u, v = state["u_uid"], state["v_uid"]
    du_s = np.zeros(u.size, np.int64); dv_s = np.zeros(v.size, np.int64)
    du_l = np.zeros(u.size, np.int64); dv_l = np.zeros(v.size, np.int64)
    if state["edges_u"].size:
        et = state["edges_type"]
        pu = np.searchsorted(u, state["edges_u"])
        pv = np.searchsorted(v, state["edges_v"])
        s = et == SHORT
        np.add.at(du_s, pu[s], 1);  np.add.at(dv_s, pv[s], 1)
        np.add.at(du_l, pu[~s], 1); np.add.at(dv_l, pv[~s], 1)
    du = du_s + du_l; dv = dv_s + dv_l
    return (np.concatenate([du, dv]),
            np.concatenate([du_s, dv_s]),
            np.concatenate([du_l, dv_l]))


def accumulate_mixing(state, Mage, Mcomm):
    """Add this month's edges to the band x band and community x community counts."""
    if not state["edges_u"].size:
        return
    u, v = state["u_uid"], state["v_uid"]
    pu = np.searchsorted(u, state["edges_u"])
    pv = np.searchsorted(v, state["edges_v"])
    np.add.at(Mage,  (state["u_band"][pu], state["v_band"][pv]), 1)
    np.add.at(Mcomm, (state["u_comm"][pu].astype(int),
                      state["v_comm"][pv].astype(int)), 1)


def pmf(counts, kmax):
    counts = np.asarray(counts, dtype=np.int64)
    h = np.bincount(counts, minlength=kmax + 1)[:kmax + 1].astype(float)
    s = h.sum()
    return h / s if s else h


def band_labels(params):
    edges = params["age_band_edges"]; lo, hi = params["age_range"]
    lows = [lo, *edges]; highs = [*edges, hi]
    return [f"{int(a)}-{int(b)}" for a, b in zip(lows, highs)]


# =====================================================================
# build + CALIBRATE
# =====================================================================
params = M.interpretable_to_params(USER_PARAMS, N_MEN, nV=N_WOMEN)
model = M.build_model(N_MEN, nV=N_WOMEN)
params = M.calibrate(model, params)          # tune rho + p_form_long to inputs
rng = np.random.default_rng(SEED)
state = M.init_network_state(model, params, rng)

MPS = model["months_per_step"]
BURN = M._default_burn_months(params)
target = USER_PARAMS["mean_partners_per_year"]
target_flong = params["frac_long_target"]
p_form_long = params["p_form_long"]
n_bands = params["n_bands"]; n_comm = params["n_communities"]
blab = band_labels(params)

# storage --------------------------------------------------------------
inst_pool, inst_short_pool, inst_long_pool = [], [], []
annual_pool = []
inst_mean_t, yearly_mean_t, standing_long_frac_t = [], [], []
durations, durations_type = [], []
Mage  = np.zeros((n_bands, n_bands))
Mcomm = np.zeros((n_comm, n_comm))

# incidence of NEW long/short ties per year, and people holding >=1 long tie
new_long_per_year, new_short_per_year = [], []
_nl_accum = _ns_accum = 0
ppl_with_long_t, ppl_with_2long_t, ppl_long_frac_t = [], [], []

# singleness: single NOW (0 current partners) and NEVER-partnered since start.
# Closed population here (no demography), so array position == a stable person id.
single_now_t, always_single_t = [], []
ever_partnered = set()
N_PEOPLE = N_MEN + N_WOMEN

# annual (reset each 12 mo) and lifetime (never reset) partner accumulators
yr_pm, yr_pw, yr_am, yr_aw = {}, {}, set(), set()
life_pm, life_pw, life_am, life_aw = {}, {}, set(), set()

# --- burn-in: only track the standing long-tie fraction (for panel 8) ---
for month in range(1, BURN + 1):
    M.network_step(state, model, params, rng, month)
    et = state["edges_type"]
    standing_long_frac_t.append((et == LONG).mean() if et.size else 0.0)

# --- measurement phase ---
# To exercise external demography instead, pass e.g.
#   births = {"age": np.full(k, 16.0), "comm": rng.integers(0, n_comm, k)}
#   M.network_step(..., extra_deaths_U=some_uids, births_U=births)
total_months = YEARS * 12
for m in range(1, total_months + 1):
    month = BURN + m
    M.network_step(state, model, params, rng, month,
                   track_durations=True,
                   durations=durations, durations_type=durations_type)

    # instantaneous degree (total / short / long)
    tot, sh, lo_ = inst_degrees_by_type(state)
    inst_pool.append(tot); inst_short_pool.append(sh); inst_long_pool.append(lo_)
    inst_mean_t.append(tot.mean())

    # incidence: ties FORMED this month (edge_birth == month), by type
    _new = state["edge_birth"] == month
    _et_new = state["edges_type"][_new]
    _nl_accum += int((_et_new == LONG).sum())
    _ns_accum += int((_et_new == SHORT).sum())
    # people currently holding >=1 (and >=2) long ties
    ppl_with_long_t.append(int((lo_ >= 1).sum()))
    ppl_with_2long_t.append(int((lo_ >= 2).sum()))
    ppl_long_frac_t.append(float((lo_ >= 1).mean()))

    # singleness: single now = 0 current partners; always-single = never partnered
    # at ANY step since measurement start (a survival curve that only decreases)
    single_now_t.append(int((tot == 0).sum()))
    ever_partnered.update(np.nonzero(tot >= 1)[0].tolist())
    always_single_t.append(N_PEOPLE - len(ever_partnered))

    # standing long fraction
    et = state["edges_type"]
    standing_long_frac_t.append((et == LONG).mean() if et.size else 0.0)

    # age / community mixing
    accumulate_mixing(state, Mage, Mcomm)

    # annual + lifetime distinct-partner accumulation
    yr_am.update(state["u_uid"].tolist()); yr_aw.update(state["v_uid"].tolist())
    life_am.update(state["u_uid"].tolist()); life_aw.update(state["v_uid"].tolist())
    for a, b in zip(state["edges_u"].tolist(), state["edges_v"].tolist()):
        yr_pm.setdefault(a, set()).add(b);   yr_pw.setdefault(b, set()).add(a)
        life_pm.setdefault(a, set()).add(b); life_pw.setdefault(b, set()).add(a)

    if m % 12 == 0:
        counts = np.array([len(yr_pm.get(u, ())) for u in yr_am]
                          + [len(yr_pw.get(v, ())) for v in yr_aw])
        yearly_mean_t.append(counts.mean())
        annual_pool.append(counts)
        new_long_per_year.append(_nl_accum)
        new_short_per_year.append(_ns_accum)
        _nl_accum = _ns_accum = 0
        yr_pm, yr_pw, yr_am, yr_aw = {}, {}, set(), set()

# pool the timescale distributions
inst_all   = np.concatenate(inst_pool)
inst_short = np.concatenate(inst_short_pool)
inst_long  = np.concatenate(inst_long_pool)
annual_all = np.concatenate(annual_pool)
lifetime   = np.array([len(life_pm.get(u, ())) for u in life_am]
                      + [len(life_pw.get(v, ())) for v in life_aw])
durations = np.array(durations, dtype=float)
durations_type = np.array(durations_type, dtype=np.int8)
new_long_per_year = np.array(new_long_per_year, dtype=float)
new_short_per_year = np.array(new_short_per_year, dtype=float)
ppl_with_long_t = np.array(ppl_with_long_t, dtype=float)
ppl_with_2long_t = np.array(ppl_with_2long_t, dtype=float)
ppl_long_frac_t = np.array(ppl_long_frac_t, dtype=float)
single_now_t = np.array(single_now_t, dtype=float)
always_single_t = np.array(always_single_t, dtype=float)

# realised mixing (row-normalised conditional distributions)
Mage_cond = Mage / Mage.sum(1, keepdims=True).clip(min=1)
Mcomm_cond = Mcomm / Mcomm.sum(1, keepdims=True).clip(min=1)

# predicted age mixing = input A weighted by partner AVAILABILITY per band,
# row-normalised (the proportionate-availability mixing) — for the recovery check
nV_band = np.bincount(state["v_band"], minlength=n_bands).astype(float)
A_pred = params["A_age"] * nV_band[None, :]
A_pred = A_pred / A_pred.sum(1, keepdims=True).clip(min=1e-12)
age_corr = np.corrcoef(Mage_cond.ravel(), A_pred.ravel())[0, 1]

# =====================================================================
# report
# =====================================================================
print("\n" + "=" * 64)
print("REALISED vs TARGET  (after calibration + burn-in)")
print("=" * 64)
print(f"mean partners/year : realised {np.mean(yearly_mean_t):.3f}   target {target:.3f}")
print(f"standing long-frac : realised {np.mean(standing_long_frac_t[BURN:]):.3f}   "
      f"target {target_flong:.3f}   (formation prob = {p_form_long:.3f})")
print(f"durations  short: {durations[durations_type==SHORT].mean():.2f} "
      f"(target {USER_PARAMS['D_mean_short']})   "
      f"long: {durations[durations_type==LONG].mean():.2f} "
      f"(target {USER_PARAMS['D_mean_long']}; censoring lowers it)")
print(f"mean degree  instantaneous {inst_all.mean():.3f}   "
      f"annual {annual_all.mean():.3f}   lifetime({YEARS}y) {lifetime.mean():.3f}")
within_comm = float(np.trace(Mcomm) / Mcomm.sum())
print(f"within-community edge fraction : {within_comm:.3f}   "
      f"(theory 1/(1+{n_comm-1}*off) = {1/(1+(n_comm-1)*USER_PARAMS['community_off_diag']):.3f})")
print(f"age-mixing recovery : corr(realised, availability-weighted A) = {age_corr:.3f}")
_lshare = new_long_per_year.mean() / max(new_long_per_year.mean() + new_short_per_year.mean(), 1)
print(f"new relationships/year : long {new_long_per_year.mean():.0f}   "
      f"short {new_short_per_year.mean():.0f}   (long share {100*_lshare:.1f}%)")
print(f"people with >=1 long tie : {ppl_with_long_t.mean():.0f} "
      f"({100*ppl_long_frac_t.mean():.1f}% of {N_MEN+N_WOMEN});   "
      f">=2 concurrent: {ppl_with_2long_t.mean():.0f}")
print(f"single NOW (0 partners) : {single_now_t.mean():.0f} "
      f"({100*single_now_t.mean()/N_PEOPLE:.1f}%);   "
      f"never partnered in {YEARS}y : {always_single_t[-1]:.0f} "
      f"({100*always_single_t[-1]/N_PEOPLE:.1f}%)")

# =====================================================================
# plots
# =====================================================================
fig, axes = plt.subplots(3, 4, figsize=(22, 15))

# 1. degree distribution on three timescales
ax = axes[0, 0]
kmax = int(max(inst_all.max(), annual_all.max(), lifetime.max()))
xk = np.arange(kmax + 1)
for arr, lab, col in [(inst_all, "instantaneous", "C0"),
                      (annual_all, "annual (12 mo)", "C2"),
                      (lifetime, f"lifetime ({YEARS} yr)", "C3")]:
    ax.plot(xk, pmf(arr, kmax), "-", lw=1.8, color=col, alpha=0.85, label=lab)
    ax.axvline(arr.mean(), ls="--", color=col, alpha=0.5)
ax.set_title("1. Degree distribution by timescale\n(instantaneous / annual / lifetime)")
ax.set_xlabel("distinct partners"); ax.set_ylabel("fraction of people")
ax.set_yscale("log"); 
ax.set_xscale("log");
ax.set_xlim(0.1, kmax + 0.5); ax.legend()

# 2. instantaneous degree split by type
ax = axes[0, 1]
kmax2 = int(inst_all.max())
xk2 = np.arange(kmax2 + 1)
for arr, lab, col in [(inst_short, "short (casual)", "C0"),
                      (inst_long, "long (steady)", "C3"),
                      (inst_all, "total", "k")]:
    ax.plot(xk2, pmf(arr, kmax2), "o-", ms=3, color=col, alpha=0.85, label=lab)
ax.set_title("2. Instantaneous degree by TYPE\n(short / long / total)")
ax.set_xlabel("current partners"); ax.set_ylabel("fraction of people")
ax.set_yscale("log"); ax.legend()

# 3. input age preference A
ax = axes[0, 2]
im = ax.imshow(params["A_age"], origin="lower", cmap="viridis", aspect="auto")
ax.set_title("3. Input age preference A\n(relative weight)")
ax.set_xlabel("woman age band"); ax.set_ylabel("man age band")
ax.set_xticks(range(n_bands)); ax.set_xticklabels(blab, rotation=45, ha="right", fontsize=7)
ax.set_yticks(range(n_bands)); ax.set_yticklabels(blab, fontsize=7)
fig.colorbar(im, ax=ax, fraction=0.046)

# 4. realised partner-age mixing (row-normalised)
ax = axes[0, 3]
im = ax.imshow(Mage_cond, origin="lower", cmap="viridis", aspect="auto", vmin=0)
ax.set_title(f"4. Realised age mixing (row-norm.)\nP(woman band | man band)  corr={age_corr:.2f}")
ax.set_xlabel("woman age band"); ax.set_ylabel("man age band")
ax.set_xticks(range(n_bands)); ax.set_xticklabels(blab, rotation=45, ha="right", fontsize=7)
ax.set_yticks(range(n_bands)); ax.set_yticklabels(blab, fontsize=7)
fig.colorbar(im, ax=ax, fraction=0.046)

# 5. realised community mixing (row-normalised)
ax = axes[1, 0]
im = ax.imshow(Mcomm_cond, origin="lower", cmap="magma", aspect="auto", vmin=0, vmax=1)
ax.set_title(f"5. Realised community mixing\n(within-comm frac = {within_comm:.2f})")
ax.set_xlabel("woman community"); ax.set_ylabel("man community")
ax.set_xticks(range(n_comm)); ax.set_yticks(range(n_comm))
for i in range(n_comm):
    for j in range(n_comm):
        ax.text(j, i, f"{Mcomm_cond[i,j]:.2f}", ha="center", va="center",
                color="w" if Mcomm_cond[i, j] < 0.5 else "k", fontsize=8)
fig.colorbar(im, ax=ax, fraction=0.046)

# 6. durations by type
ax = axes[1, 1]
dmax = int(durations.max())
bins = np.arange(0, dmax + 2) - 0.5
ax.hist(durations[durations_type == SHORT], bins=bins, density=True, histtype="step",
        lw=2, color="C0", label="short (casual)")
ax.hist(durations[durations_type == LONG], bins=bins, density=True, histtype="step",
        lw=2, color="C3", label="long (steady)")
ax.axvline(USER_PARAMS["D_mean_short"], ls="--", color="C0", alpha=0.6)
ax.axvline(USER_PARAMS["D_mean_long"], ls="--", color="C3", alpha=0.6)
ax.set_title("6. Realised durations by TYPE")
ax.set_xlabel("duration (months)"); ax.set_ylabel("fraction of partnerships")
ax.set_yscale("log"); ax.legend()

# 7. calibration: mean partners/year over time vs target
ax = axes[1, 2]
yy = np.arange(1, YEARS + 1)
ax.axhline(target, ls="--", color="gray", label="target input")
ax.plot(yy, yearly_mean_t, "o-", color="C2", label="realised (calibrated)")
ax.set_ylim(0, max(max(yearly_mean_t), target) * 1.3)
ax.set_title("7. Mean partners/year vs target")
ax.set_xlabel("measurement year"); ax.set_ylabel("mean distinct partners / year")
ax.legend()

# 8. long-tie fraction over time
ax = axes[1, 3]
mm = np.arange(1, len(standing_long_frac_t) + 1)
ax.plot(mm, standing_long_frac_t, color="C3", lw=1.0, label="standing (realised)")
ax.axhline(target_flong, ls="--", color="gray", label=f"target standing = {target_flong}")
ax.axhline(p_form_long, ls=":", color="C0", label=f"formation prob = {p_form_long:.2f}")
ax.axvline(BURN, color="k", lw=0.8, alpha=0.4)
ax.text(BURN, 0.02, " end of burn-in", fontsize=8, alpha=0.6)
ax.set_ylim(0, 1)
ax.set_title("8. Long-tie fraction over time")
ax.set_xlabel("month (incl. burn-in)"); ax.set_ylabel("fraction of edges that are long")
ax.legend(loc="center right", fontsize=8)

# 9. new LONG relationships formed per year (incidence, whole population)
ax = axes[2, 0]
yy = np.arange(1, YEARS + 1)
ax.bar(yy, new_long_per_year, color="C3", alpha=0.85, label="new long / yr")
ax.axhline(new_long_per_year.mean(), ls="--", color="k",
           label=f"mean {new_long_per_year.mean():.0f}/yr")
ax.set_title("9. NEW long-term relationships / year\n(formed across whole population)")
ax.set_xlabel("measurement year"); ax.set_ylabel("new long partnerships formed")
ax.set_ylim(0, new_long_per_year.max() * 1.2 if new_long_per_year.size else 1)
ax.legend(fontsize=8)

# 10. people holding >=1 (and >=2) long relationships over time
ax = axes[2, 1]
mm2 = np.arange(1, len(ppl_with_long_t) + 1)
ax.plot(mm2, ppl_with_long_t, color="C3", lw=1.0, label=">=1 long tie")
ax.plot(mm2, ppl_with_2long_t, color="C1", lw=1.0, alpha=0.85, label=">=2 (concurrent)")
ax.axhline(ppl_with_long_t.mean(), ls="--", color="k",
           label=f"mean {ppl_with_long_t.mean():.0f}")
ax.set_title(f"10. People with >=1 long relationship\n"
             f"({100*ppl_long_frac_t.mean():.1f}% of N={N_MEN+N_WOMEN})")
ax.set_xlabel("measurement month"); ax.set_ylabel("number of people")
ax.legend(loc="center right", fontsize=8)

# 11. singleness: single now vs never-partnered-since-start
ax = axes[2, 2]
mm3 = np.arange(1, len(single_now_t) + 1)
ax.plot(mm3, single_now_t, color="C0", lw=1.2,
        label=f"single NOW (0 partners), mean {single_now_t.mean():.0f}")
ax.plot(mm3, always_single_t, color="C3", lw=1.6,
        label=f"never partnered since start ({100*always_single_t[-1]/N_PEOPLE:.0f}% at end)")
ax.axhline(single_now_t.mean(), ls="--", color="C0", alpha=0.4)
ax.set_ylim(0, N_PEOPLE)
ax.set_title("11. Singleness over time\n(single-at-instant vs never-partnered)")
ax.set_xlabel("measurement month"); ax.set_ylabel(f"number of people (N={N_PEOPLE})")
ax.legend(loc="upper right", fontsize=7)

axes[2, 3].axis("off")   # unused slot in the 3x4 grid

fig.suptitle(
    f"Age + community bipartite network (calibrated)  —  N={N_MEN}+{N_WOMEN}, "
    f"partners/yr={target}, {n_comm} communities, {n_bands} age bands, "
    f"D_short={USER_PARAMS['D_mean_short']}/D_long={USER_PARAMS['D_mean_long']} mo, "
    f"standing long-frac={target_flong}",
    fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT_PNG, dpi=120)
print(f"\nsaved figure -> {OUT_PNG}")
