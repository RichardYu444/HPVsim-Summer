"""
r0_corrected_acts.py
====================

How many sex acts per relationship -- and what per-act transmissibility -- put R0 between 1 and
1.5 for every genotype of run_r0_corrected_1900.py (hpv16, hpv18, hi5) on its mean-degree-1.4
network (NETWORK=gamma2, powerlaw3p5 or default; everything here follows that script's NETWORK). Run
through that script, in this order:

    --network     records the partnership timelines (and draws the network figures)
    --r0-truth    simulated ground-truth R0 (section 4); GT_BETAS / GT_SEEDS / GT_LABEL add rounds
    --acts        analytic scans + ground truth -> the choice, csvs/<TAG>_acts_analysis.txt, figure


1. One partnership
------------------
HPVsim transmits within a partnership, on each quarterly step, with probability

    P = 1 - (1 - p)^(a dt)            a = acts per year in that partnership (age-scaled)
    p = beta * rel_beta_g * trans_dir * (1 - condoms_layer * eff_condoms)
        trans_dir = 1 (female->male) or 3.69 (male->female)

(whole acts plus a fractional one -- r0_hpv.step_prob). Over an overlap of n steps between the
partnership and an infectious period, the chance it transmits is

    T = 1 - (1 - p)^(a dt n) = 1 - exp(-lambda * overlap),     lambda = a * (-ln(1 - p))

so acts and per-act probability enter ONLY through the hazard lambda. Doubling acts is exactly
doubling lambda; halving p is (for small p) halving lambda. "Needed acts" at a given p is
therefore a* = lambda* / (-ln(1 - p)), where lambda* is the hazard that gives the target R0.


2. From partnerships to R0
--------------------------
r0_hpv.py's two-sex, two-type next-generation matrix (type A = infected by a partner they already
had, type B = infected at the start of a new partnership):

    R0 = sqrt(R_m R_f),   R_s = sum_tau [ S_tau E T_cur(lambda) + C_tau E T_new(lambda) ]
    T_cur = lambda/(lambda+q) (1 - e^{-(lambda+q) D})                    to a current partner
    T_new = lambda/(lambda+q) (D - (1 - e^{-(lambda+q) D})/(lambda+q))   per new partner per year

S_tau = other current partners, C_tau = new partners per year, q_tau = dissolution rate, D the
infectious period. As lambda -> infinity, T -> 1 and R_s -> S + C E[D]: the network alone caps R0.
This DISTINCT-person R0 is r0_hpv's convention.

HPVsim also lets a partner who has cleared be infected again by the same person (men never gain
natural immunity; women's is partial), so a long partnership can pass the infection back and
forth. Counting every infection a case causes -- reff_hpv.py's "all-events" R0, whose crossing of
1 is what decides whether a genotype grows or declines in HPVsim -- adds, for each partner k,
renewal cycles: wait ~Exp(lambda') to infection, k infectious for D_k, k susceptible again (women
with their new immunity), repeat until the overlap ends.


3. Evaluating it
----------------
Both are evaluated on the ACTUAL recorded partnership timelines rather than on idealised constant
rates (the constant-rate closed form overshoots -- r0_hpv.py's validation table):

    distinct     r0_hpv.estimate_r0 'window' (validated against one-generation outbreaks)
    all-events   reff_hpv.expected_offspring with snap=None -- immune-naive, no competition
                 (verified to 5.5% median error against realised offspring, run_reff_verification)

Two scans: acts scaled by s at the current beta (lambda scales by exactly s), and beta scaled at
the current acts.

On the Gamma-2 network the two BRACKET the truth rather than hit it: distinct-person R0 stays below
1 at any number of acts (~0.8 at saturation), while the reinfection-counting one (~1.9 at saturation)
over-counts -- it treats every back-and-forth reinfection inside a couple as a new, independent
chain with an average case's offspring, whereas in a mostly monogamous, sparse network those chains
are closed loops. (Its 5.5% verification was of R_eff at equilibrium on the default network, where
concurrency is higher.)


4. Ground truth, and the choice
-------------------------------
ground_truth() (``--r0-truth``) measures R0 directly: small-seed outbreaks in an immune-naive
population, every transmission logged (reff_hpv.TxLog), and for each case infected in GT_WINDOW the
number of transmissions its infection episode actually made -- reinfections included, with every
HPVsim mechanism (immunity on clearance, partnership ends, death, competition) in play. Per sex
R_s = mean offspring, R0 = sqrt(R_f R_m), which is the per-generation growth factor of the two-sex
process (the mean offspring over the stable case mix is the dominant eigenvalue). A direct growth
test agrees with it (grows at beta 0.25, declines at 0.0344).

main() then chooses the beta that centres all three genotypes in [1, 1.5] on the ground-truth curve
(maximises the smaller of min R0 - 1 and 1.5 - max R0), and maps it to the acts that would give the
same R0 at the current beta, through section 1's equivalence as the analytic estimator applies it.

The measurement only reads R0 while the outbreak is still small during GT_WINDOW. On the default
network a fast-growing outbreak saturates inside the window at beta >= 0.05 (~50k-230k hpv16 cases in
100k agents, against <= 17k at every beta on the Gamma-2 / power-law networks), so its realised
offspring fall with beta there -- that is R_eff under depletion, not R0. GT_FIT_MAX_BETA (env) keeps
those points out of the fit and the choice; they stay in the table and the figure, hollow.
"""
import os
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sciris as sc

import r0_hpv
import reff_hpv
import run_r0_corrected_1900 as run


R0_LO, R0_HI = 1.0, 1.5
N_REP = 5              # infectious-period draws per index case
MAX_EVENTS = 40_000    # uniform subsample of index events per scan point (full set for the final checks)
SEED = 1               # common random numbers across scan points, so curves are smooth

ACTS_SCALES = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0, 5.0, 100.0]
BETAS = [0.25, 0.2, 0.15, 0.1, 0.07, 0.05, 0.04, 0.03, 0.025, 0.02, 0.015, 0.01, 0.005]

# colours: each genotype keeps the colour plot_powerlaw_nogate.GENOTYPES gives it in every other
# figure of the project (reference palette slots 2-4); text tokens from the same palette
GENO_COLOR = {'hpv16': '#eb6834', 'hpv18': '#1baf7a', 'hi5': '#eda100'}
GENO_MARKER = {'hpv16': 'o', 'hpv18': 's', 'hi5': '^'}
INK, INK_2, INK_MUTED, SURFACE, BAND = '#0b0b0b', '#52514e', '#898781', '#fcfcfb', '#e7e6e1'


# -------------------------------------------------------------------
# R0 at one (acts scale, beta)
# -------------------------------------------------------------------

class Evaluator:
    """ Holds the recorded network and the index events; evaluates R0 at any (acts scale, beta) """

    def __init__(self, ctx, max_events=MAX_EVENTS):
        self.ctx = ctx
        self.rec = ctx['rec']
        self.base_pars = sc.dcp(ctx['pars'])
        self.base_acts = self.rec.acts.copy()
        self.W = int(round(1 / self.base_pars['dt']))
        ev = reff_hpv.uniform_events(ctx, self.rec.t0, self.W)
        self.n_events_total = len(ev)
        if max_events and len(ev) > max_events:
            keep = np.sort(np.random.default_rng(0).choice(len(ev), max_events, replace=False))
            ev = ev.iloc[keep].reset_index(drop=True)
            ev['ev'] = np.arange(len(ev))
        self.ev = ev
        self.genotypes = dict(self.base_pars['genotype_map'])

    def set(self, acts_scale=1.0, beta=None):
        self.ctx['pars'] = sc.dcp(self.base_pars)
        self.ctx['pars']['beta'] = self.base_pars['beta'] if beta is None else beta
        self.rec.acts = self.base_acts * acts_scale

    def reset(self):
        self.set()

    def r0(self, acts_scale=1.0, beta=None, n_rep=N_REP):
        """ {genotype: dict(all, dist, R_m, R_f)} -- all-events and distinct from the same draws """
        self.set(acts_scale, beta)
        out = {}
        try:
            for g, gname in self.genotypes.items():
                np.random.seed(SEED + g)
                ev = self.ev.copy()
                ev['infector_avail'] = reff_hpv.naive_infector_avail(self.ctx, g, ev, None)
                off = reff_hpv.expected_offspring(self.ctx, g, ev, None, n_rep=n_rep)
                ks = reff_hpv.k_sums(ev, off)
                ra = reff_hpv.r_from_sums(ks['all'], ks['n'])
                rd = reff_hpv.r_from_sums(ks['dist'], ks['n'])
                out[gname] = dict(all=ra['rho'], dist=rd['rho'], R_m=ra['R_m'], R_f=ra['R_f'],
                                  R_m_dist=rd['R_m'], R_f_dist=rd['R_f'])
        finally:
            self.reset()
        return out

    def r0_window(self, acts_scale=1.0, beta=None, n_rep=N_REP):
        """ r0_hpv's validated distinct-person 'window' estimator (all events, not subsampled) """
        self.set(acts_scale, beta)
        try:
            np.random.seed(SEED)
            est, stats = r0_hpv.estimate_r0(self.rec, self.ctx['pars'], n_rep=n_rep)
        finally:
            self.reset()
        return {g: v['window']['R0'] for g, v in est.items()}, stats


# -------------------------------------------------------------------
# scans and the solve
# -------------------------------------------------------------------

def scan(ev_obj, kind, values):
    rows = []
    for v in values:
        T = sc.tic()
        res = ev_obj.r0(acts_scale=v, beta=None) if kind == 'acts' else ev_obj.r0(acts_scale=1.0, beta=v)
        for g, r in res.items():
            rows.append(dict(scan=kind, value=v, genotype=g, **r))
        print(f"  {kind:>4} {v:>7g}  " + '  '.join(f"{g} {r['all']:.3f}/{r['dist']:.3f}" for g, r in res.items())
              + f"   ({sc.toc(T, output=True):.0f}s)", flush=True)
    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# reporting
# -------------------------------------------------------------------

def per_act_table(pars, beta):
    """ Per-act p and per-act hazard -ln(1-p), by genotype, layer and direction """
    rows = []
    for g in run.VAX_GENOTYPES:
        gi = [k for k, v in pars['genotype_map'].items() if v == g][0]
        for lkey in (run.LKEY_SHORT, run.LKEY_LONG):
            for direction in ('m2f', 'f2m'):
                q = dict(pars, beta=beta)
                p = r0_hpv.per_act_prob(q, gi, lkey, direction)
                rows.append(dict(genotype=g, layer=lkey, direction=direction, p=p,
                                 hazard_per_act=-np.log1p(-min(p, 1 - 1e-12))))
    return pd.DataFrame(rows)


def load_ground_truth():
    """
    Every per-seed ground-truth CSV written so far (the first round and any GT_LABEL rounds),
    pooled per (beta, genotype): R_s = case-weighted mean offspring of sex s, R = sqrt(R_f R_m).
    Returns (pooled, raw) or (None, None) if --r0-truth has not been run.
    """
    files = sorted(p for p in run.CSV_DIR.glob(f'{run.TAG}_r0_ground_truth*.csv') if 'pooled' not in p.name)
    if not files:
        return None, None
    raw = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    raw = raw.drop_duplicates(['beta', 'seed', 'genotype'], keep='last')
    rows = []
    for (b, g), d in raw.groupby(['beta', 'genotype']):
        row = dict(beta=b, genotype=g, n_seeds=len(d), n_cases=int(d['n_cases'].sum()))
        for lab in ('all', 'dist'):
            Rf = np.average(d[f'R_f_{lab}'], weights=d['n_f']) if d['n_f'].sum() else np.nan
            Rm = np.average(d[f'R_m_{lab}'], weights=d['n_m']) if d['n_m'].sum() else np.nan
            row[f'R_{lab}'] = float(np.sqrt(Rf * Rm))
        row['R_all_lo'], row['R_all_hi'] = float(d['R_all'].min()), float(d['R_all'].max())
        rows.append(row)
    return pd.DataFrame(rows), raw


def interp_curves(df, xcol, ycol, fine):
    """ Each genotype's ycol against xcol, piecewise-linear in log x, on the grid `fine` """
    out = {}
    for g, d in df.groupby('genotype'):
        d = d.sort_values(xcol)
        out[g] = np.interp(np.log(fine), np.log(d[xcol].to_numpy()), d[ycol].to_numpy())
    return out


def fit_curves(gt, fine):
    """
    Each genotype's simulated R0 as a smooth function of beta: case-weighted least squares of a
    quadratic in log beta (enough to bend into the saturation), evaluated on `fine`. Choosing on the
    fit rather than on the joined points keeps seed noise at any one beta from moving the choice.
    """
    out = {}
    for g, d in gt.groupby('genotype'):
        c = np.polyfit(np.log(d['beta']), d['R_all'], 2, w=np.sqrt(d['n_cases']))
        out[g] = np.polyval(c, np.log(fine))
    return out


def round_beta(beta):
    """ A round number near the maximin: to 0.005 where R0 is flat in beta, to 0.001 below 0.05 where it is not """
    step = 0.005 if beta >= 0.05 else 0.001
    return float(np.round(np.round(beta / step) * step, 6))


def maximin(curves, fine):
    """ The x on `fine` that centres every genotype in [R0_LO, R0_HI]; (x, margin, {g: R}) """
    C = np.array(list(curves.values()))
    margin = np.minimum(C.min(axis=0) - R0_LO, R0_HI - C.max(axis=0))
    i = int(np.argmax(margin))
    return float(fine[i]), float(margin[i]), {g: float(c[i]) for g, c in curves.items()}


def acts_equivalent(scan_df, beta_star):
    """
    The acts scale (at the current beta) with the same analytic reinfection-counting R0 as beta_star
    at the current acts, per genotype, geometric mean over genotypes. This is the acts <-> per-act p
    mapping of section 1, taken through the analytic estimator (which is biased in level but applies
    the same partnership-level transmission either way).
    """
    fine_b = np.array([beta_star])
    target = interp_curves(scan_df[scan_df['scan'] == 'beta'].rename(columns={'value': 'x'}), 'x', 'all', fine_b)
    a = scan_df[(scan_df['scan'] == 'acts') & (scan_df['value'] <= 5.0)]
    s = []
    for g, d in a.groupby('genotype'):
        d = d.sort_values('value')
        y = np.maximum.accumulate(d['all'].to_numpy())  # monotone for inversion (MC noise near saturation)
        s.append(np.exp(np.interp(target[g][0], y, np.log(d['value'].to_numpy()))))
    return float(np.exp(np.mean(np.log(s)))), s


def plot(scan_df, gt, cur, beta_star, s_star, out_png, gt_out=None):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4), facecolor=SURFACE)

    def frame(ax, title, xlabel):
        ax.set_facecolor(SURFACE)
        ax.axhspan(R0_LO, R0_HI, color=BAND, zorder=0, label=f'target R0 {R0_LO:g}-{R0_HI:g}')
        ax.axhline(1.0, color=INK_MUTED, lw=0.8, zorder=1)
        ax.set_xscale('log')
        ax.set_ylim(0, 2.3)
        ax.set_xlabel(xlabel, color=INK_2)
        ax.set_ylabel('R0', color=INK_2)
        ax.set_title(title, color=INK, fontsize=11, loc='left')
        ax.grid(True, which='major', color='#ecebe7', lw=0.6)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        for sp in ('left', 'bottom'):
            ax.spines[sp].set_color(INK_MUTED)
        ax.tick_params(colors=INK_2)

    def vlines(ax, xc, xs, lab):
        ax.axvline(xc, color=INK_2, lw=1, ls=':')
        ax.text(xc, 2.25, ' current', color=INK_2, fontsize=9, va='top')
        ax.axvline(xs, color=INK, lw=1.2)
        ax.text(xs, 2.25, f' chosen\n {lab}', color=INK, fontsize=9, va='top')

    # A: per-act transmissibility, acts fixed -- ground truth between the two analytic bounds
    ax = axes[0]
    frame(ax, f'R0 against per-act transmissibility, acts fixed ({cur["acts_s"]:g} / {cur["acts_l"]:g} a year at peak)',
          'beta  (per-act transmission probability, hpv16 female->male, no condom)')
    b = scan_df[(scan_df['scan'] == 'beta')]
    for g, color in GENO_COLOR.items():
        dg = b[b['genotype'] == g].sort_values('value')
        ax.plot(dg['value'], dg['all'], ':', color=color, lw=1.4, alpha=0.9)
        ax.plot(dg['value'], dg['dist'], '--', color=color, lw=1.2, alpha=0.8)
        if gt is not None:
            t = gt[gt['genotype'] == g].sort_values('beta')
            # markers differ by genotype: orange/yellow are too close to rely on colour alone
            ax.errorbar(t['beta'], t['R_all'], yerr=[t['R_all'] - t['R_all_lo'], t['R_all_hi'] - t['R_all']],
                        color=color, ls='none', marker=GENO_MARKER[g], ms=6, capsize=2, elinewidth=0.8)
            if gt_out is not None and len(gt_out):
                o = gt_out[gt_out['genotype'] == g].sort_values('beta')
                ax.plot(o['beta'], o['R_all'], ls='none', marker=GENO_MARKER[g], ms=6, mfc='none', mec=color)
            xf = np.exp(np.linspace(np.log(t['beta'].min()), np.log(t['beta'].max()), 200))
            ax.plot(xf, fit_curves(gt[gt['genotype'] == g], xf)[g], '-', color=color, lw=2,
                    marker=GENO_MARKER[g], markevery=[0], ms=6,
                    label=f'{g}: simulated (realised offspring): pooled, seed range, fit')
    if gt_out is not None and len(gt_out):
        ax.plot([], [], ls='none', marker='o', ms=6, mfc='none', mec=INK_2,
                label=f'hollow: beta > {GT_FIT_MAX_BETA:g}, saturated outbreak (R_eff), not fitted')
    vlines(ax, cur['beta'], beta_star, f'{beta_star:.3g}')

    # B: acts, per-act transmissibility fixed -- analytic bounds, chosen = the acts-equivalent of beta*
    ax = axes[1]
    frame(ax, f'R0 against acts, per-act transmissibility fixed (beta = {cur["beta"]:g}); analytic bounds only',
          'mean acts per partnership-year (age-scaled, both layers pooled)')
    a = scan_df[(scan_df['scan'] == 'acts') & (scan_df['value'] <= 5.0)]
    for g, color in GENO_COLOR.items():
        dg = a[a['genotype'] == g].sort_values('value')
        x = dg['value'] * cur['mean_acts']
        ax.plot(x, dg['all'], ':', color=color, lw=1.4, alpha=0.9, label=f'{g}: analytic, counting reinfection (upper)')
        ax.plot(x, dg['dist'], '--', color=color, lw=1.2, alpha=0.8, label=f'{g}: analytic, distinct people (lower)')
    vlines(ax, cur['mean_acts'], s_star * cur['mean_acts'], f'{s_star * cur["mean_acts"]:.3g}')

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh); labels.append(ll)
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=8.5, frameon=False)
    fig.suptitle(f'{run.NET_LABEL}, mean degree {run.MEAN_DEGREE} (excl. singles): R0 by genotype\n'
                 'simulated ground truth (solid) between the two analytic estimators (dotted upper, dashed lower)',
                 color=INK, fontsize=12)
    fig.tight_layout(rect=[0, 0.12, 1, 0.93])
    fig.savefig(out_png, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    print(f'saved {out_png}')


def main():
    ctx = sc.load(run.ctx_path())
    rec = ctx['rec']
    pars = ctx['pars']
    dt = pars['dt']
    lkeys = list(rec.lkeys)
    cur = dict(beta=float(pars['beta']), acts_s=float(pars['acts'][run.LKEY_SHORT]['par1']),
               acts_l=float(pars['acts'][run.LKEY_LONG]['par1']))
    mean_acts_layer = {lk: float(rec.acts[rec.layer == i].mean()) for i, lk in enumerate(lkeys)}
    cur['mean_acts'] = float(rec.acts.mean())
    net = ctx.get('net', {})
    S, L = run.LKEY_SHORT, run.LKEY_LONG
    dur_mo = {S: float(np.mean(net['dur_short'])) if len(net.get('dur_short', [])) else np.nan,
              L: float(np.mean(net['dur_long'])) if len(net.get('dur_long', [])) else np.nan}
    print(f"Recorded network: {len(rec.eid):,} partnerships from step {rec.t0} to {rec.t_end}; "
          f"mean acts/yr {S} {mean_acts_layer[S]:.1f}, {L} {mean_acts_layer[L]:.1f}", flush=True)

    ev = Evaluator(ctx)
    print(f'{ev.n_events_total:,} uniform index events, {len(ev.ev):,} used per scan point', flush=True)

    # --- distinct-person R0 by the validated window estimator, now and saturated ---
    win_now, stats = ev.r0_window()
    win_sat, _ = ev.r0_window(acts_scale=100.0)
    print('window R0 now', {g: round(v, 3) for g, v in win_now.items()},
          ' saturated', {g: round(v, 3) for g, v in win_sat.items()}, flush=True)

    # --- analytic scans (cached: ACTS_RESCAN=1 recomputes) ---
    scan_csv = run.CSV_DIR / f'{run.TAG}_acts_scan.csv'
    if scan_csv.exists() and os.environ.get('ACTS_RESCAN', '0') != '1':
        scan_df = pd.read_csv(scan_csv)
        print(f'analytic scan loaded from {scan_csv.name}', flush=True)
    else:
        print('\nScan (all-events / distinct):', flush=True)
        scan_df = pd.concat([scan(ev, 'acts', ACTS_SCALES), scan(ev, 'beta', BETAS)], ignore_index=True)
        scan_df.to_csv(scan_csv, index=False)
        print(f'wrote {scan_csv}')

    # --- ground truth and the choice ---
    gt, gt_raw = load_ground_truth()
    if gt is None:
        raise FileNotFoundError('no ground truth yet -- run run_r0_corrected_1900.py --r0-truth first')
    gt.to_csv(run.CSV_DIR / f'{run.TAG}_r0_ground_truth_pooled_all.csv', index=False)
    gt_all, gt_out = gt, gt[gt['beta'] > GT_FIT_MAX_BETA]
    gt = gt[gt['beta'] <= GT_FIT_MAX_BETA]
    fine = np.exp(np.linspace(np.log(gt['beta'].min()), np.log(gt['beta'].max()), 4001))
    beta_raw, margin, _ = maximin(fit_curves(gt, fine), fine)
    beta_star = round_beta(beta_raw)
    _, _, gt_at_star = maximin(fit_curves(gt, np.array([beta_star])), np.array([beta_star]))
    s_star, s_by_g = acts_equivalent(scan_df, beta_star)
    an_star = ev.r0(beta=beta_star)
    an_s = ev.r0(acts_scale=s_star)
    print(f'chosen beta {beta_star} (maximin {beta_raw:.4f}, margin {margin:.3f}); acts-equivalent scale {s_star:.3f}',
          flush=True)

    plot(scan_df, gt, cur, beta_star, s_star, run.FIG_DIR / f'{run.TAG}_acts_R0.png', gt_out=gt_out)

    # --- report ---
    pt_now = per_act_table(pars, cur['beta'])
    pt_new = per_act_table(pars, beta_star)
    E_D = {}
    for gi, g in pars['genotype_map'].items():
        E_D[g] = {sex: float(r0_hpv.sample_transmitting_steps(pars, gi, sex, 100_000).mean() * dt)
                  for sex in ('m', 'f')}
    an_now = ev.r0()
    an_sat = ev.r0(acts_scale=100.0)

    def fmt_an(r):
        return '  '.join(f"{g} {v['dist']:.2f}-{v['all']:.2f}" for g, v in r.items())

    tab = gt_all.pivot(index='beta', columns='genotype', values='R_all')[list(run.VAX_GENOTYPES)]
    nse = gt_all.pivot(index='beta', columns='genotype', values='n_seeds')[list(run.VAX_GENOTYPES)].min(axis=1)
    ncase = gt_all.pivot(index='beta', columns='genotype', values='n_cases')['hpv16'] / nse
    L = [
        f'{run.TAG}  --  acts per relationship / per-act transmissibility for R0 in [{R0_LO}, {R0_HI}]',
        '',
        f"Network: {run.NET_LABEL}, mean degree {run.MEAN_DEGREE} excl. singles, recorded "
        f"{ctx['net']['yearvec'][rec.t0]:.0f}-{ctx['net']['yearvec'][-1]:.0f}",
        f"  other current partners at infection S (m/f): "
        f"{sum(stats['m']['A']['S'].values()):.3f} / {sum(stats['f']['A']['S'].values()):.3f};  "
        f"new partners per year C (m/f): "
        f"{sum(stats['m']['A']['C'].values()):.3f} / {sum(stats['f']['A']['C'].values()):.3f}",
        '  dissolution q per year: ' + ', '.join(f'{k} {v:.3f}' for k, v in stats['q'].items())
        + f';  completed lengths {run.TYPE_LABEL[S]} {dur_mo[S]:.1f} mo, {run.TYPE_LABEL[L]} {dur_mo[L]:.1f} mo (censored run)',
        '  mean transmitting period E[D] (years): ' + '; '.join(
            f"{g} m {d['m']:.2f} f {d['f']:.2f}" for g, d in E_D.items()),
        f"  current transmission: beta {cur['beta']:g}, acts/yr at peak {cur['acts_s']:g} / {cur['acts_l']:g} "
        f"(realised mean {mean_acts_layer[S]:.1f} / {mean_acts_layer[L]:.1f} a partnership-year)",
        '',
        '1. ANALYTIC (partnership timelines, section 2) -- two bounds, as distinct-people / counting-reinfection:',
        '   now        ' + fmt_an(an_now),
        '   saturated  ' + fmt_an(an_sat) + '   (acts x100)',
        f"   Saturated distinct-person R0 is {max(v['dist'] for v in an_sat.values()):.2f} at most"
        + (' -- below 1 at ANY number of acts.' if max(v['dist'] for v in an_sat.values()) < 1 else '.'),
        '   The reinfection-counting one over-counts, treating each back-and-forth reinfection inside a',
        '   couple as a new, independent chain; the simulated R0 below lies between the two.',
        '',
        '2. SIMULATED GROUND TRUTH -- realised offspring (every later transmission, reinfections included)',
        f'   of cases infected {GT_WINDOW[0]}-{GT_WINDOW[1]} in small-seed outbreaks in an immune-naive',
        f'   population, R = sqrt(R_f R_m), pooled over seeds (case-weighted):',
        '   ' + tab.round(3).assign(seeds=nse.astype(int), hpv16_cases_per_seed=ncase.round(0).astype(int))
        .to_string().replace('\n', '\n   '),
        '   This is the per-generation growth factor, which decides persistence (on the Gamma-2 network a',
        '   direct growth test agreed: beta 0.25 grew, 0.0344 declined). Highest simulated value per genotype: '
        + ', '.join(f'{g} {tab[g].max():.2f}' for g in tab.columns) + '.',
        '',
        *([f'   Fitted on beta <= {GT_FIT_MAX_BETA:g} only: above it the outbreak saturates inside the case window',
           '   (see the case counts), so realised offspring there are R_eff under depletion, not R0.']
          if len(gt_out) else []),
        '',
        f'3. CHOICE -- beta centring all three in [{R0_LO}, {R0_HI}] on the fitted curves (maximin {beta_raw:.4f}, rounded):',
        f'   beta {cur["beta"]:g} -> {beta_star:g} (x{beta_star / cur["beta"]:.3f}), acts unchanged at '
        f'{cur["acts_s"]:g} / {cur["acts_l"]:g} a year at peak',
        '   simulated R0 at the choice (fit): ' + '  '.join(f'{g} {v:.2f}' for g, v in gt_at_star.items()),
        '   analytic bounds there:      ' + fmt_an(an_star),
        '',
        '   Needed acts, per-act transmissibility unchanged (the same R0 through section 1\'s equivalence,',
        '   taken via the analytic estimator):',
        f'   acts x{s_star:.3f} -> {cur["acts_s"] * s_star:.1f} / {cur["acts_l"] * s_star:.1f} a year at peak; realised '
        f'{mean_acts_layer[S] * s_star:.1f} / {mean_acts_layer[L] * s_star:.1f} a partnership-year;',
        f'   about {mean_acts_layer[S] * s_star * dur_mo[S] / 12:.0f} acts in a {run.TYPE_LABEL[S]} relationship and '
        f'{mean_acts_layer[L] * s_star * dur_mo[L] / 12:.0f} in a {run.TYPE_LABEL[L]} one (x mean completed length)',
        '   analytic bounds there:      ' + fmt_an(an_s) + '   (by genotype: x' + ', x'.join(f'{v:.3f}' for v in s_by_g) + ')',
        '',
        'Per-act transmission probability p (hazard -ln(1-p)), now -> chosen:',
    ]
    for (_, a), (_, b) in zip(pt_now.iterrows(), pt_new.iterrows()):
        L.append(f"  {a['genotype']:>5} {a['layer']} {a['direction']}:  p {a['p']:.3f} -> {b['p']:.4f}   "
                 f"hazard/act {a['hazard_per_act']:.3f} -> {b['hazard_per_act']:.4f}")
    L += [
        '',
        f"rel_beta (unchanged): { {g: pars['genotype_pars'][g]['rel_beta'] for g in run.VAX_GENOTYPES} }",
        'Equivalence: T = 1 - exp(-a (-ln(1-p)) overlap), so (acts, p) pairs with the same per-year hazard',
        'a (-ln(1-p)) give the same R0 (up to HPVsim rounding acts per quarter). Scaling acts cuts every hazard',
        'by the same factor; scaling beta cuts male->female (large p) hazards by more than female->male.',
    ]
    text = '\n'.join(L) + '\n'
    out = run.CSV_DIR / f'{run.TAG}_acts_analysis.txt'
    out.write_text(text, encoding='utf-8')
    print('\n' + text)
    print(f'wrote {out}')
    sol = dict(beta_star=beta_star, beta_raw=beta_raw, margin=margin, gt_at_star=gt_at_star, s_star=s_star,
               s_by_g=s_by_g, an_star=an_star, an_s=an_s, an_now=an_now, an_sat=an_sat, cur=cur, stats=stats,
               win_now=win_now, win_sat=win_sat)
    sc.save(run.CSV_DIR / f'{run.TAG}_acts_solution.obj', sol)
    return sol


# -------------------------------------------------------------------
# ground truth: realised offspring in small-seed outbreaks
# -------------------------------------------------------------------

# env-overridable (comma lists) so a refinement round can add seeds/betas; GT_LABEL suffixes its CSVs
GT_BETAS = [float(x) for x in os.environ.get('GT_BETAS', '0.25,0.15,0.1,0.07,0.05,0.0344').split(',')]
GT_SEEDS = [int(x) for x in os.environ.get('GT_SEEDS', '0,1,2').split(',')]
GT_LABEL = os.environ.get('GT_LABEL', '')
GT_FIT_MAX_BETA = float(os.environ.get('GT_FIT_MAX_BETA', 'inf'))  # see section 4
GT_END = 1950
GT_SEED_FRAC = 0.1                # seeded prevalence, as a fraction of the usual initial prevalence
GT_WINDOW = (1903, 1913)          # cases infected in these years; their offspring counted to GT_END


def _gt_sim(beta, seed):
    """
    One outbreak: immune-naive population, a tenth of the usual initial prevalence, no
    interventions, every transmission logged. Returns the TxLog frame and the step window.
    """
    import hpvsim_working as hpv
    pars = run.make_pars(seed, end=GT_END, interventions=[], beta=beta, verbose=-1,
                         rel_init_prev=run.init_prev_keep_fraction() * GT_SEED_FRAC,
                         track_transmission=True, analyzers=[reff_hpv.TxLog()])
    sim = hpv.Sim(pars)
    sim.run()
    yv = np.asarray(sim.yearvec)
    steps = (int(np.searchsorted(yv, GT_WINDOW[0])), int(np.searchsorted(yv, GT_WINDOW[1])))
    female = np.asarray(sim.people.is_female, bool)
    return dict(beta=beta, seed=seed, tx=sim.get_analyzer('TxLog').df, steps=steps, female=female,
                genotypes=dict(sim['genotype_map']))


def realised_r(res):
    """
    Mean realised offspring of the cases infected in GT_WINDOW -- every later transmission by the
    same infection episode, reinfections of the same partner included (all-events) and at most one
    per partner (distinct) -- per sex, and R = sqrt(R_m R_f), the per-generation growth factor of
    the two-sex process (as r0_hpv's R0).
    """
    tx, (s0, s1), fem = res['tx'], res['steps'], res['female']
    rows = []
    for g, gname in res['genotypes'].items():
        t = tx[tx['g'] == g]
        cases = t[(t['t'] >= s0) & (t['t'] < s1)][['tgt', 't', 'src']].drop_duplicates(['tgt', 't'])
        cases = cases.rename(columns={'tgt': 'person', 't': 'start', 'src': 'infector'})
        off = t[['src', 'src_start', 'tgt']].rename(columns={'src': 'person', 'src_start': 'start'})
        m = cases.merge(off, on=['person', 'start'], how='left')
        m['hit'] = m['tgt'].notna()
        n_all = m.groupby(['person', 'start'])['hit'].sum()
        d = m[m['hit'] & (m['tgt'] != m['infector'])].drop_duplicates(['person', 'start', 'tgt'])
        n_dist = d.groupby(['person', 'start']).size().reindex(n_all.index, fill_value=0)
        sex_f = fem[n_all.index.get_level_values('person')]
        out = dict(beta=res['beta'], seed=res['seed'], genotype=gname, n_cases=len(n_all),
                   n_f=int(sex_f.sum()), n_m=int((~sex_f).sum()))
        for lab, n in (('all', n_all), ('dist', n_dist)):
            Rf = float(n[sex_f].mean()) if sex_f.any() else np.nan
            Rm = float(n[~sex_f].mean()) if (~sex_f).any() else np.nan
            out.update({f'R_f_{lab}': Rf, f'R_m_{lab}': Rm, f'R_{lab}': float(np.sqrt(Rf * Rm))})
        rows.append(out)
    return pd.DataFrame(rows)


def ground_truth(ncpus=6):
    cfgs = [dict(beta=b, seed=s) for b in GT_BETAS for s in GT_SEEDS]
    print(f'Ground truth: {len(cfgs)} outbreaks, {run.N_AGENTS:,} agents, {run.START}-{GT_END}, '
          f'cases infected {GT_WINDOW[0]}-{GT_WINDOW[1]}', flush=True)
    results = sc.parallelize(_gt_sim, iterkwargs=cfgs, ncpus=ncpus)
    df = pd.concat([realised_r(r) for r in results], ignore_index=True)
    out = run.CSV_DIR / f'{run.TAG}_r0_ground_truth{GT_LABEL}.csv'
    df.to_csv(out, index=False)
    # pooled over seeds, weighting by cases
    agg = []
    for (b, g), d in df.groupby(['beta', 'genotype']):
        w_f, w_m = d['n_f'].to_numpy(float), d['n_m'].to_numpy(float)
        row = dict(beta=b, genotype=g, n_cases=int(d['n_cases'].sum()))
        for lab in ('all', 'dist'):
            Rf = np.average(d[f'R_f_{lab}'], weights=w_f) if w_f.sum() else np.nan
            Rm = np.average(d[f'R_m_{lab}'], weights=w_m) if w_m.sum() else np.nan
            row[f'R_{lab}'] = float(np.sqrt(Rf * Rm))
        agg.append(row)
    agg = pd.DataFrame(agg)
    agg.to_csv(run.CSV_DIR / f'{run.TAG}_r0_ground_truth_pooled{GT_LABEL}.csv', index=False)
    print(agg.pivot(index='beta', columns='genotype', values='R_all').round(3).to_string())
    print(f'wrote {out}')
    return agg


if __name__ == '__main__':
    main()
