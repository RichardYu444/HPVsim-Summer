"""
network.py -- cumulative partnership-network plots for each network model.

Same thing the old snapshot version tracked -- every partnership that ever existed, as a
bipartite female/male graph, exported to node/edge CSVs and drawn -- but read off
hpv.network_history() instead of yearly hpv.snapshot()s, and for every network model:

    default         basePars.py (HPVsim default network, calibrated poisson)
    gamma<k>        basePars_community.py with community_pars['gamma_shape'] = k, for each
                    shape in run_sim_gamma_sweep.py's sweep plus basePars_community's own 2
    powerlaw3       powerlaw.make_sim() -- Pareto propensity, alpha = 3 (powerlaw.COMMUNITY_PARS,
                    the same network as the alpha=3 core-vacc / attribution runs)

What changed vs the snapshot version:
  * Exact. Yearly snapshots missed every partnership that formed and dissolved between two
    snapshot dates (most short partnerships); the delta stream has all of them.
  * No year window by default: the graph is the union of every partnership over the whole run
    (pass --window 2035 2040 to get the old behaviour back).
  * Each edge keeps its eid, layer and start/end timestep. The graph is an nx.MultiGraph keyed
    on eid, since the default network's casual layer can hold two simultaneous partnerships
    between the same pair (see degree_timeseries.py).
  * Network only: no infections are seeded, no interventions, one genotype -- the epidemic
    never runs, so the network is identical to the one the full model would build.

Run from the repo root with the summerhpvsim env:

    python network.py                         # all models at N_AGENTS, grid + per-model PNGs + CSVs
    python network.py --n_agents 2000         # a different population size
    python network.py --sweep 500 1000 2000 5000 --seeds 3
                                              # population-size test: a grid per size, plus
                                              # network_popsize_check.png comparing the graph
                                              # statistics across sizes and seeds

Importing powerlaw monkeypatches the community network's theta sampler for the whole process;
this module captures the Gamma original first and swaps the right one in around each sim (see
_theta_sampler), so all models can share one process.
"""
import argparse
import contextlib
import pathlib
import time

import numpy as np
import pandas as pd
import sciris as sc
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

import hpvsim_working as hpv
from hpvsim_working import age_community_bipartite_network_model as acbnm
from hpvsim_working import community_network as hpcn

_GAMMA_THETA = acbnm._sample_side_theta  # must be captured before powerlaw is imported
import powerlaw  # installs the Pareto sampler as an import side effect
_PARETO_THETA = acbnm._sample_side_theta
acbnm._sample_side_theta = hpcn._sample_side_theta = _GAMMA_THETA  # Gamma unless asked otherwise

import basePars
import basePars_community


ROOT = pathlib.Path(__file__).parent
FIG_DIR = ROOT / 'network_plots'
CSV_DIR = ROOT / 'csvs' / 'network_graphs'

N_AGENTS = 1000   # chosen from the --sweep population-size test, see main()'s docstring
SEED = 0
END_YEAR = 2040   # sims run from each base-pars start year (1980) to here
WINDOW = None     # (first_year, last_year) to restrict to partnerships active in that window; None = whole run
N_CPUS = 5

GAMMA_SHAPES = [0.05, 0.25, 1.0, 2.0, 2.5, 5.0]  # run_sim_gamma_sweep.py's sweep + basePars_community's 2
POWERLAW_ALPHA = powerlaw.COMMUNITY_PARS['gamma_shape']

LONG_LAYERS = {'m', 'l'}  # default 'm' (marital) / community 'l' (long); 'c' and 's' are short

# Reference data-viz palette: categorical slots 1-3 (validated all-pairs) + neutral ink
COLOR_F = '#eb6834'
COLOR_M = '#2a78d6'
COLOR_SHORT = '#1baf7a'
COLOR_LONG = '#52514e'
INK = '#0b0b0b'
INK_2 = '#52514e'
INK_MUTED = '#898781'
SURFACE = '#fcfcfb'


def model_keys():
    return ['default'] + [f'gamma{g:g}' for g in GAMMA_SHAPES] + [f'powerlaw{POWERLAW_ALPHA:g}']


def model_label(key):
    if key == 'default':
        return 'Default network'
    if key.startswith('gamma'):
        return f"Gamma, shape {key[len('gamma'):]}"
    return f"Power law, α = {key[len('powerlaw'):]}"


# -------------------------------------------------------------------
# Network-only sims
# -------------------------------------------------------------------

def _network_only(pars):
    ''' Strip the epidemic: one genotype, nothing seeded, no interventions or multiscale clones '''
    g = 'hpv16'
    genotype_pars = sc.objdict({g: sc.dcp(pars['genotype_pars'][g])})
    return dict(genotypes=[g], genotype_pars=genotype_pars, init_hpv_dist={g: 1.0},
                rel_init_prev=0.0, interventions=[], ms_agent_ratio=1, verbose=0,
                analyzers=[hpv.network_history()])


@contextlib.contextmanager
def _theta_sampler(pareto):
    sampler = _PARETO_THETA if pareto else _GAMMA_THETA
    acbnm._sample_side_theta = hpcn._sample_side_theta = sampler
    try:
        yield
    finally:
        acbnm._sample_side_theta = hpcn._sample_side_theta = _GAMMA_THETA


def make_sim(key, n_agents, seed):
    common = dict(n_agents=n_agents, rand_seed=seed, end=END_YEAR)
    if key == 'default':
        pars = sc.dcp(basePars.base_pars_geno)
    elif key.startswith('gamma'):
        pars = sc.dcp(basePars_community.base_pars_geno)
        pars['community_pars']['gamma_shape'] = float(key[len('gamma'):])
    else:
        # powerlaw.make_sim() builds the Sim itself, merging these on top of its own pars
        return powerlaw.make_sim(**common, **_network_only(powerlaw.base_pars),
                                 community_pars=sc.mergedicts(powerlaw.COMMUNITY_PARS,
                                                              dict(gamma_shape=POWERLAW_ALPHA)))
    pars.update(common)
    pars.update(_network_only(pars))
    return hpv.Sim(pars)


def run_network(key, n_agents, seed, window=None):
    ''' Run one network-only sim; returns (edges DataFrame, nodes DataFrame, sim info dict) '''
    T = time.time()
    with _theta_sampler(pareto=key.startswith('powerlaw')):
        sim = make_sim(key, n_agents, seed)
        sim.run()
    nh = sim.get_analyzer('network_history')
    t0, t1 = window_steps(sim, window)
    edges = collect_edges(nh, t0, t1)
    edges['start_year'] = step_year(sim, edges['t_start'])
    edges['end_year'] = step_year(sim, edges['t_end'])

    ppl = sim.people
    uids = np.unique(np.concatenate([edges['f'].values, edges['m'].values]))
    nodes = pd.DataFrame(dict(
        uid=uids,
        sex=np.where(ppl.sex[uids] == 0, 'f', 'm'),
        bipartite=np.where(ppl.sex[uids] == 0, 1, 0),  # old CSVs: males 0, females 1
        community=ppl.community[uids].astype(int),
        alive_at_end=ppl.alive[uids].astype(bool),
        age_at_end=np.round(ppl.age[uids], 2),
    ))
    info = dict(model=key, n_agents=n_agents, seed=seed, t0=t0, t1=t1,
                n_ever_alive=len(ppl), run_seconds=time.time() - T)
    return edges, nodes, info


def window_steps(sim, window):
    if window is None:
        return 0, sim.npts - 1
    yv = np.asarray(sim.yearvec)
    return int(np.searchsorted(yv, window[0])), int(np.searchsorted(yv, window[1], side='right') - 1)


def step_year(sim, t):
    ''' Year at the start of timestep t; -1 (formed before the sim) -> start year; NaN stays NaN '''
    t = np.asarray(t, dtype=float)
    out = np.full(t.shape, np.nan)
    ok = ~np.isnan(t)
    out[ok] = np.asarray(sim.yearvec)[np.clip(t[ok], 0, sim.npts - 1).astype(int)]
    return out


def collect_edges(nh, t0, t1):
    '''
    Every partnership active at any point in timesteps t0..t1, one row per eid, from a single
    replay of the delta stream (adds before removals, as in network_history.edges_at()).
    t_start = -1 for partnerships in the initial snapshot; t_end = NaN if still active at t1.
    '''
    lm = nh.layer_map
    live = {}
    rows = []
    d = nh.initial_snapshot
    for eid, f, m, layer in zip(d.added_edges.eid, d.added_edges.f, d.added_edges.m, d.added_edges.layer):
        live[int(eid)] = (int(f), int(m), lm[layer], -1)
    for t in sorted(nh.deltas):
        if t > t1:
            break
        delta = nh.deltas[t]
        for eid, f, m, layer in zip(delta.added_edges.eid, delta.added_edges.f,
                                    delta.added_edges.m, delta.added_edges.layer):
            live[int(eid)] = (int(f), int(m), lm[layer], t)
        for eid in delta.removed_edges.eid:
            rec = live.pop(int(eid), None)
            if rec is not None and t >= t0:
                rows.append((int(eid), *rec, t))
    rows += [(eid, *rec, np.nan) for eid, rec in live.items()]
    edges = pd.DataFrame(rows, columns=['eid', 'f', 'm', 'layer', 't_start', 't_end'])
    edges['long'] = edges['layer'].isin(LONG_LAYERS)
    return edges.sort_values('eid').reset_index(drop=True)


def build_graph(edges, nodes):
    G = nx.MultiGraph()
    for row in nodes.itertuples(index=False):
        G.add_node(int(row.uid), **row._asdict())
    for row in edges.itertuples(index=False):
        G.add_edge(int(row.f), int(row.m), key=int(row.eid), layer=row.layer, long=bool(row.long),
                   start_year=row.start_year, end_year=row.end_year)
    return G


def graph_stats(G):
    deg = np.array([d for _, d in G.degree()], dtype=float)
    comp_sizes = np.array(sorted((len(c) for c in nx.connected_components(G)), reverse=True))
    n = G.number_of_nodes()
    long_edges = sum(1 for *_, lg in G.edges(data='long') if lg)
    top10 = np.sort(deg)[::-1][:max(1, int(round(0.1 * n)))]
    return dict(
        n_nodes=n,
        n_edges=G.number_of_edges(),
        frac_long=long_edges / max(G.number_of_edges(), 1),
        mean_degree=deg.mean(),
        cv_degree=deg.std() / deg.mean(),
        max_degree=int(deg.max()),
        top10_edge_share=top10.sum() / deg.sum(),  # share of partnership-ends held by the top 10%
        n_components=len(comp_sizes),
        lcc_frac=comp_sizes[0] / n,
        dyad_frac=2 * (comp_sizes == 2).sum() / n,  # people whose only partner(s) had no one else
    )


# -------------------------------------------------------------------
# Drawing
# -------------------------------------------------------------------

def spread_out(xy):
    '''
    Radial rank-equalisation of a layout: keep each node's direction from the centre and the
    ORDER of distances, but move node i to radius sqrt(rank_i / n), which gives uniform density
    per unit area. The spring layout of a graph with ~15-35 partners per person packs almost
    everyone into a thin dense core; this opens the core up without changing who sits next to
    whom (increasing spring_layout's k was tried and barely moved it).
    '''
    d = xy - np.median(xy, axis=0)
    r = np.hypot(d[:, 0], d[:, 1])
    rank = (np.argsort(np.argsort(r)) + 0.5) / len(r)
    return d / np.maximum(r, 1e-12)[:, None] * np.sqrt(rank)[:, None]


def packed_layout(G, seed=0):
    '''
    Lay each connected component out on its own, then shelf-pack the components largest-first.
    A single spring layout of a graph this fragmented (hundreds of dyads next to one big
    component) flings the small pieces to the edges and crushes the big one into the middle.
    '''
    S = nx.Graph(G)  # layout only needs the simple graph
    comps = sorted(nx.connected_components(S), key=len, reverse=True)
    boxes = []
    for comp in comps:
        size = len(comp)
        side = np.sqrt(size) + 1.0
        if size == 2:
            a, b = comp
            local = {a: np.array([-0.35, 0.0]), b: np.array([0.35, 0.0])}
        else:
            sub = S.subgraph(comp)
            local = nx.spring_layout(sub, seed=seed, iterations=100 if size > 500 else 200)
            xy = spread_out(np.array(list(local.values())))
            xy /= max(np.abs(xy).max(), 1e-9)
            local = {u: p * (side / 2 - 0.5) for u, p in zip(local.keys(), xy)}
        boxes.append((side, local))

    width = max(np.sqrt(sum(s * s for s, _ in boxes)) * 1.05, boxes[0][0])
    pos = {}
    x = y = row_h = 0.0
    for side, local in boxes:
        if x + side > width and x > 0:
            x, y, row_h = 0.0, y - row_h, 0.0
        centre = np.array([x + side / 2, y - side / 2])
        for u, p in local.items():
            pos[u] = centre + p
        x += side
        row_h = max(row_h, side)
    return pos


def draw_network(ax, G, pos, title, stats):
    ax.set_facecolor(SURFACE)
    n = G.number_of_nodes()
    # Fade edges as their number grows, so dense regions read as density rather than solid ink
    alpha = float(np.clip(25.0 / np.sqrt(max(G.number_of_edges(), 1)), 0.06, 0.8))
    for is_long, color, lw in ((False, COLOR_SHORT, 0.5), (True, COLOR_LONG, 0.6)):
        segs = [(pos[u], pos[v]) for u, v, lg in G.edges(data='long') if lg == is_long]
        if segs:
            ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, alpha=alpha, zorder=1))

    # Lowest degree first so hubs sit on top; sexes interleaved chunk by chunk so neither
    # colour buries the other (one scatter per sex would paint all of one over the other)
    base = float(np.clip(1500.0 / n, 1.5, 14.0))
    order = sorted(G.nodes(), key=G.degree)
    for chunk in np.array_split(np.array(order), min(20, n)):
        for sex, color, marker in (('f', COLOR_F, 'o'), ('m', COLOR_M, 's')):
            us = [u for u in chunk if G.nodes[u]['sex'] == sex]
            if not us:
                continue
            xy = np.array([pos[u] for u in us])
            size = base * np.sqrt(np.array([G.degree(u) for u in us], dtype=float))
            ax.scatter(xy[:, 0], xy[:, 1], s=size, c=color, marker=marker, linewidths=0, zorder=2)

    ax.autoscale_view()
    ax.set_aspect('equal', adjustable='datalim')  # 'box' would shrink the axes and shift the titles
    ax.axis('off')
    ax.set_title(title, fontsize=11, color=INK, loc='left', pad=16)
    ax.text(0.0, 1.0,
            f"{stats['n_nodes']:,} people · {stats['n_edges']:,} partnerships · "
            f"largest component {stats['lcc_frac']:.0%} · max {stats['max_degree']} partners",
            transform=ax.transAxes, fontsize=8, color=INK_2, va='bottom')


def legend_handles():
    return [
        Line2D([], [], marker='o', ls='', color=COLOR_F, markersize=6, label='female'),
        Line2D([], [], marker='s', ls='', color=COLOR_M, markersize=6, label='male'),
        Line2D([], [], color=COLOR_LONG, lw=1.5, label='long / marital partnership'),
        Line2D([], [], color=COLOR_SHORT, lw=1.5, label='short / casual partnership'),
    ]


def span_text(window):
    return 'whole run' if window is None else f'{window[0]}–{window[1]}'


def plot_grid(results, n_agents, seed, window, out_png):
    keys = [k for k in model_keys() if k in results]
    ncol = 4
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 6.4 * nrow), facecolor=SURFACE)
    for ax in np.ravel(axes)[len(keys):]:
        ax.axis('off')
    for ax, key in zip(np.ravel(axes), keys):
        G, pos, stats = results[key]
        draw_network(ax, G, pos, model_label(key), stats)
    fig.legend(handles=legend_handles(), loc='lower center', ncol=4, frameon=False, fontsize=10)
    fig.suptitle(f'Cumulative partnership network ({span_text(window)}, sims run to {END_YEAR}) -- '
                 f'n_agents = {n_agents:,}, seed {seed}. Components packed largest first; '
                 f'node area ∝ √(partners)', fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_png, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    print(f'saved {out_png}')


def plot_single(key, G, pos, stats, n_agents, seed, window, out_png):
    fig, ax = plt.subplots(figsize=(10, 10.6), facecolor=SURFACE)
    draw_network(ax, G, pos, f'{model_label(key)} -- n_agents = {n_agents:,}, seed {seed}, {span_text(window)}', stats)
    ax.legend(handles=legend_handles(), loc='upper right', frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, facecolor=SURFACE)
    plt.close(fig)


# -------------------------------------------------------------------
# Drivers
# -------------------------------------------------------------------

def _task(key, n_agents, seed, want_graph, window):
    edges, nodes, info = run_network(key, n_agents, seed, window)
    G = build_graph(edges, nodes)
    stats = {**info, **graph_stats(G)}
    out = dict(stats=stats)
    if want_graph:
        out.update(edges=edges, nodes=nodes, pos=packed_layout(G, seed=seed))
    print(f"  {key:>12}  n={n_agents:>6}  seed={seed}  {stats['n_nodes']:>6} people  "
          f"{stats['n_edges']:>6} edges  LCC {stats['lcc_frac']:.2f}  ({stats['run_seconds']:.0f}s)")
    return out


def run_all(n_list, seeds, window, csv, draw_max=None):
    ''' draw_max: sizes above this are statistics-only (no layout, plots or CSVs) '''
    tasks = [dict(key=k, n_agents=n, seed=s, window=window,
                  want_graph=(s == seeds[0] and (draw_max is None or n <= draw_max)))
             for n in n_list for s in seeds for k in model_keys()]
    outs = sc.parallelize(_task, iterkwargs=tasks, ncpus=N_CPUS)

    FIG_DIR.mkdir(exist_ok=True)
    for n in n_list:
        if draw_max is not None and n > draw_max:
            continue
        results = {}
        for task, out in zip(tasks, outs):
            if task['n_agents'] != n or not task['want_graph']:
                continue
            key = task['key']
            G = build_graph(out['edges'], out['nodes'])
            results[key] = (G, out['pos'], out['stats'])
            if csv:
                CSV_DIR.mkdir(parents=True, exist_ok=True)
                tag = f'{key}_n{n}_seed{seeds[0]}'
                out['nodes'].to_csv(CSV_DIR / f'{tag}_nodes.csv', index=False)
                out['edges'].to_csv(CSV_DIR / f'{tag}_edges.csv', index=False)
            plot_single(key, G, out['pos'], out['stats'], n, seeds[0], window,
                        FIG_DIR / f'network_{key}_n{n}.png')
        plot_grid(results, n, seeds[0], window, FIG_DIR / f'network_grid_n{n}.png')
    return pd.DataFrame([out['stats'] for out in outs])


POPSIZE_METRICS = [
    ('mean_degree', 'Mean partners per person'),
    ('cv_degree', 'CV of partners per person'),
    ('top10_edge_share', 'Share of partnerships held\nby the most active 10%'),
    ('lcc_frac', 'Share of people in the\nlargest component'),
    ('max_degree', 'Most partners held by\nany one person'),
    ('n_nodes', 'People drawn (clutter)'),
]


def plot_popsize_check(df, out_png):
    ''' Each metric vs n_agents (mean over seeds, with min-max range), one line per model '''
    keys = [k for k in model_keys() if k in set(df['model'])]
    gamma_keys = [k for k in keys if k.startswith('gamma')]
    ramp = ['#86b6ef', '#5598e7', '#2a78d6', '#256abf', '#184f95', '#0d366b']  # ordinal blue, step 250 up
    colors = {k: ramp[round(i * (len(ramp) - 1) / max(len(gamma_keys) - 1, 1))] for i, k in enumerate(gamma_keys)}
    colors.update({k: INK_2 for k in keys if k == 'default'})
    colors.update({k: COLOR_F for k in keys if k.startswith('powerlaw')})

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5), facecolor=SURFACE)
    for ax, (col, label) in zip(axes.ravel(), POPSIZE_METRICS):
        ax.set_facecolor(SURFACE)
        for key in keys:
            g = df[df['model'] == key].groupby('n_agents')[col]
            x = g.mean().index.values
            ax.fill_between(x, g.min().values, g.max().values, color=colors[key], alpha=0.12, linewidth=0)
            ax.plot(x, g.mean().values, '-o', color=colors[key], lw=2, ms=4, label=model_label(key))
        ax.set_xscale('log')
        ax.set_xticks(sorted(df['n_agents'].unique()))
        ax.set_xticklabels([f'{n:,}' for n in sorted(df['n_agents'].unique())])
        ax.minorticks_off()
        ax.set_title(label, fontsize=11, color=INK, loc='left')
        ax.set_xlabel('n_agents', color=INK_2)
        ax.grid(True, color='#e1e0d9', lw=0.6)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color('#c3c2b7')
        ax.tick_params(colors=INK_MUTED)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle(f'Population-size check: cumulative-network statistics vs n_agents '
                 f"(line = mean over {df['seed'].nunique()} seeds, band = min–max)",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    print(f'saved {out_png}')


def main():
    '''
    Population-size choice, from `--sweep 250 500 1000 2000 4000 --seeds 3`
    (csvs/network_graphs/popsize_check.csv, network_plots/network_popsize_check.png):

      * Over the whole run every model is one giant component (98-100% of people) at every
        size, and each person has ~13-35 lifetime partners whatever n is -- so a bigger
        population never thins the picture out, it only adds more of the same band.
      * Per-person statistics (mean/CV of partners, the top 10%'s share) are within seed noise
        of the 4,000-agent values from 500 agents for the default network and gamma >= 0.25.
        The power law needs ~1,000 (mean partners 30 at 500, 34 at 1,000, 35 at 2,000-4,000);
        its biggest hubs keep growing with n regardless (max partners 240 / 340 / 690 / 720).
      * Gamma 0.05 never settles in this range -- mean partners and CV still climbing at 4,000
        -- because only ~20% of people ever partner and the hubs are capped by population size.
      * Visually 500 still shows individual hubs and the sex mix, 1,000 is the limit, and
        2,000+ is a solid blob.

    Hence N_AGENTS = 1000: the smallest size at which everything except gamma 0.05 is
    representative. Use --n_agents 500 for a cleaner picture of the thinner-tailed models.
    '''
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_agents', type=int, default=N_AGENTS)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--sweep', type=int, nargs='+', default=None,
                    help='population-size test: run every model at each of these n_agents')
    ap.add_argument('--seeds', type=int, default=3, help='seeds per size in --sweep')
    ap.add_argument('--draw_max', type=int, default=2000,
                    help='in --sweep, sizes above this are statistics-only (layout gets slow)')
    ap.add_argument('--window', type=int, nargs=2, default=None, metavar=('FIRST', 'LAST'),
                    help='only partnerships active between these years (default: whole run)')
    ap.add_argument('--no_csv', action='store_true', help='skip the node/edge CSV export')
    args = ap.parse_args()
    window = tuple(args.window) if args.window else WINDOW

    if args.sweep:
        seeds = list(range(args.seed, args.seed + args.seeds))
        df = run_all(sorted(args.sweep), seeds, window, csv=not args.no_csv, draw_max=args.draw_max)
        CSV_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(CSV_DIR / 'popsize_check.csv', index=False)
        with pd.option_context('display.width', 200, 'display.max_columns', 20):
            print(df.groupby(['model', 'n_agents'])[[c for c, _ in POPSIZE_METRICS]].mean().round(3))
        plot_popsize_check(df, FIG_DIR / 'network_popsize_check.png')
    else:
        run_all([args.n_agents], [args.seed], window, csv=not args.no_csv)


if __name__ == '__main__':
    main()
