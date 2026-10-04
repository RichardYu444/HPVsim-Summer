"""
Diagnostics for two completely isolated communities on the community network: is the epidemic
in each community really its own, and are the by-community results recorded correctly?

Two communities, 'Large' (80%) and 'Small' (20%), with a diagonal community_mixing (the same
construction as run_sim_community_2iso.py), no interventions, 1980 start.

Experiments
    both_t0     both communities seeded at t=0 with the usual init_hpv_prev (what the real runs do)
    delayed     Large seeded in 1980, Small left uninfected and seeded in 2000
    low_small   both seeded in 1980, Small at 5% of the usual initial prevalence

Seeding is done by SeedCommunity below, which infects one community's members using the sim's
own init_hpv_prev (by age/sex) and init_hpv_dist (by genotype), i.e. what init_states() does at
t=0 -- rel_init_prev is set to 0 so init_states() itself infects nobody.

Every run records, at every timestep (Recorder below):
    - transmissions whose source and target are in DIFFERENT communities (needs track_transmission)
    - live contacts in people.contacts that join two different communities
    - prevalence by community, recounted directly from people with the same people.scale
      weighting as sim.results
and at the end prints the three checks: cross-community transmissions, recount vs
sim.results['hpv_prevalence_by_community'], and sim.results vs the to_df() export.

Findings (2026-09-24, 20k agents, seeds 1-2): 0 cross-community transmissions or contacts in
any run; the recount matches sim.results to ~1e-7 and the export matches exactly; Small seeded
in 2000 runs its own epidemic, a 20-year-delayed replay of Large's 1980 curve. Plot with
plot_community_isolation.py.

Usage
    python diagnose_community_isolation.py                              # the default set below
    python diagnose_community_isolation.py <experiment> <seed> <n_agents> <end>
Output: OUTPUT_DIR/diag_<experiment>_s<seed>_n<n_agents>_<end>.npz
"""
import sys, pathlib, time
import numpy as np
import sciris as sc

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import hpvsim_working as hpv
from basePars_community import base_pars_geno

OUTPUT_DIR = ROOT / 'csvs' / 'isolation_diagnostics'

# (experiment, seed, n_agents, end) run when no command-line arguments are given
DEFAULT_RUNS = [('both_t0', 1, 20_000, 2045),
                ('delayed', 1, 20_000, 2045),
                ('delayed', 2, 20_000, 2045),
                ('low_small', 1, 20_000, 2045)]

PROBS = np.array([0.8, 0.2])
LABELS = ['Large', 'Small']


def output_path(exp, seed, n_agents, end):
    return OUTPUT_DIR / f'diag_{exp}_s{seed}_n{n_agents}_{end}.npz'


def community_pars():
    ''' Two communities, diagonal mixing: no partnership can form between them '''
    pars = dict(base_pars_geno['community_pars'])
    pars.update(n_communities=2, community_probs=PROBS,
                community_mixing=np.diag(PROBS) / np.outer(PROBS, PROBS),
                community_labels=LABELS)
    return pars


class SeedCommunity:
    '''
    Intervention: at `year`, infect members of community `comm` using the sim's own init_hpv_prev
    (age/sex, scaled by `rel`) and init_hpv_dist (genotype) -- what init_states() does at t=0.
    '''

    def __init__(self, year, comm, rel=1.0):
        self.year, self.comm, self.rel, self.done = year, comm, rel, False

    def __call__(self, sim):
        if self.done or sim.yearvec[sim.t] < self.year:
            return
        self.done = True
        p = sim.people
        ip = sim['init_hpv_prev']
        brackets = np.asarray(ip['age_brackets'])
        age_inds = np.digitize(p.age, brackets)
        probs = np.zeros(len(p))
        f, m = p.is_female, p.is_male
        probs[f] = np.asarray(ip['f'])[np.minimum(age_inds[f], len(brackets) - 1)]
        probs[m] = np.asarray(ip['m'])[np.minimum(age_inds[m], len(brackets) - 1)]
        probs *= self.rel
        member = p.alive & p.is_active & (p.community == self.comm)
        probs[~member] = 0
        inds = np.nonzero(np.random.random(len(p)) < probs)[0]
        dist = np.array(list(sim['init_hpv_dist'].values()), dtype=float)
        gts = np.random.choice(len(dist), size=len(inds), p=dist / dist.sum())
        for g in range(len(dist)):
            gi = inds[gts == g]
            gi = gi[~p.infectious[g, gi]]
            p.infect(inds=gi, g=g, layer='seed_infection')
        print(f'  [seed] year {sim.yearvec[sim.t]:.2f}: infected {len(inds)} members of {LABELS[self.comm]}')


class Recorder:
    ''' Analyzer, called at the end of every step (after transmission) '''

    def __init__(self):
        self.t, self.year, self.prev_w, self.n_members = [], [], [], []
        self.cross_tx, self.within_tx, self.cross_edges, self.tx_unknown = [], [], [], []

    def __call__(self, sim):
        p = sim.people
        comm = p.community
        alive = p.alive
        prev_w = np.zeros(2); nmem = np.zeros(2)
        for c in range(2):
            mem = alive & (comm == c)
            nmem[c] = mem.sum()
            if nmem[c]:
                inf_any = p.infectious[:, mem].any(axis=0)
                w = p.scale[mem]
                prev_w[c] = (inf_any * w).sum() / w.sum()  # Same weighting as sim.results
        # Who infected whom this step (sim.step fills people.new_transmissions when
        # track_transmission is on, and clears it after the analyzers have run)
        cross = within = unknown = 0
        for src, tgt, _scale, g, lkey in p.new_transmissions:
            cs, ct = comm[src], comm[tgt]
            ok = (cs >= 0) & (ct >= 0)
            unknown += int((~ok).sum())
            cross += int((cs[ok] != ct[ok]).sum())
            within += int((cs[ok] == ct[ok]).sum())
        ce = 0
        for lkey, layer in p.contacts.items():
            if len(layer['f']):
                ce += int((comm[layer['f']] != comm[layer['m']]).sum())
        self.t.append(sim.t); self.year.append(sim.yearvec[sim.t])
        self.prev_w.append(prev_w); self.n_members.append(nmem)
        self.cross_tx.append(cross); self.within_tx.append(within)
        self.cross_edges.append(ce); self.tx_unknown.append(unknown)


def run_experiment(exp, seed, n_agents, end):
    pars = sc.dcp(base_pars_geno)
    pars.update(n_agents=n_agents, start=1980, end=end, rand_seed=seed, verbose=-1,
                interventions=[], track_transmission=True)
    pars['community_pars'] = community_pars()
    pars['community_results'] = ['hpv_prevalence', 'within_edge_frac']

    if exp == 'both_t0':
        ivs = []
    elif exp == 'delayed':
        pars['rel_init_prev'] = 0.0
        ivs = [SeedCommunity(1980, 0), SeedCommunity(2000, 1)]
    elif exp == 'low_small':
        pars['rel_init_prev'] = 0.0
        ivs = [SeedCommunity(1980, 0), SeedCommunity(1980, 1, rel=0.05)]
    else:
        raise ValueError(f'Unknown experiment {exp!r}')
    pars['interventions'] = ivs
    pars['analyzers'] = [Recorder()]

    sim = hpv.Sim(pars, label=exp)
    T = time.time()
    sim.run()
    rec = [a for a in sim.analyzers if isinstance(a, Recorder)][0]  # The sim holds a copy
    print(f'{exp} seed {seed}: ran in {time.time() - T:.0f}s')

    # Recording checks. Stocks are taken on the last timestep of each year, so compare the
    # recount on those steps only
    t = np.array(rec.t)
    prev_w = np.array(rec.prev_w)
    res_prev = sim.results['hpv_prevalence_by_community'].values
    stock_steps = t % sim.resfreq == sim.resfreq - 1
    n = stock_steps.sum()
    df = sim.to_df(date_index=True, by_community=['hpv_prevalence'])
    csv_prev = np.vstack([df[f'hpv_prevalence_by_community_{l}'].to_numpy() for l in LABELS])

    out = output_path(exp, seed, n_agents, end)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, t=t, year=np.array(rec.year), prev_w=prev_w, n_members=np.array(rec.n_members),
             cross_tx=np.array(rec.cross_tx), within_tx=np.array(rec.within_tx),
             cross_edges=np.array(rec.cross_edges), tx_unknown=np.array(rec.tx_unknown),
             resfreq=sim.resfreq, res_prev=res_prev, res_year=sim.results['year'],
             csv_prev=csv_prev, csv_year=df.index.to_numpy())

    print(f'  cross-community transmissions: {int(np.sum(rec.cross_tx))}  within: '
          f'{int(np.sum(rec.within_tx))}  (endpoints with no community: {int(np.sum(rec.tx_unknown))})')
    print(f'  most cross-community live contacts at any step: {int(np.max(rec.cross_edges))}')
    print(f'  max |recount - sim.results|: {np.abs(prev_w[stock_steps].T - res_prev[:, :n]).max():.1e}')
    print(f'  max |sim.results - to_df export|: {np.abs(res_prev - csv_prev).max():.1e}')
    print(f'  saved {out}')
    return out


if __name__ == '__main__':
    if len(sys.argv) > 1:
        exp, seed, n_agents, end = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        run_experiment(exp, seed, n_agents, end)
    else:
        for args in DEFAULT_RUNS:
            run_experiment(*args)
