"""
make_network_schematics.py
==========================

Small schematic partnership networks for the graphical abstract: the same number of people and
the same number of partnerships (so the same mean degree) drawn three ways, from homogeneous to
heavy-tailed partner propensity. These are illustrations, not model output: the real whole-run
networks (network.py, n=1000) are too dense to read at graphical-abstract size.

Colours match network.py (women orange circles, men blue squares).

Usage
-----
    python Graphical_Abstract/make_network_schematics.py
Writes Graphical_Abstract/img/net_{default,gamma,powerlaw}.png (transparent, 960 x 960 px).
"""
import pathlib

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

OUT = pathlib.Path(__file__).resolve().parent / 'img'

COLOR_F = '#eb6834'
COLOR_M = '#2a78d6'
COLOR_EDGE = '#52514e'

N_PER_SEX = 16
N_EDGES = 26  # same in every panel -> same mean degree (26 * 2 / 32 = 1.6)

# (name, propensity sampler, seed). Seeds picked so the max degree rises left to right.
NETWORKS = [
    ('default', lambda rng, n: np.ones(n), 0),
    ('gamma', lambda rng, n: rng.gamma(1.0, 1.0, n), 0),
    ('powerlaw', lambda rng, n: rng.pareto(1.3, n) + 1, 1),
]


def sample_network(propensity, seed):
    """Bipartite graph with exactly N_EDGES distinct partnerships, ends drawn by propensity."""
    rng = np.random.default_rng(seed)
    wf = propensity(rng, N_PER_SEX)
    wm = propensity(rng, N_PER_SEX)
    pf, pm = wf / wf.sum(), wm / wm.sum()
    g = nx.Graph()
    g.add_nodes_from((f'f{i}', {'sex': 'f'}) for i in range(N_PER_SEX))
    g.add_nodes_from((f'm{i}', {'sex': 'm'}) for i in range(N_PER_SEX))
    while g.number_of_edges() < N_EDGES:
        g.add_edge(f'f{rng.choice(N_PER_SEX, p=pf)}', f'm{rng.choice(N_PER_SEX, p=pm)}')
    return g


def layout(g, seed):
    """Spring layout for people with a partner, with a hidden anchor weakly tied to each so
    separate components stay in frame; people with no partner spaced evenly on an outer ring."""
    active = [n for n in g if g.degree(n) > 0]
    isolates = [n for n in g if g.degree(n) == 0]
    h = nx.Graph()
    h.add_edges_from(g.edges, weight=1.0)
    h.add_edges_from((('anchor', n) for n in active), weight=0.04)
    pos = nx.spring_layout(h, seed=seed, k=0.55, iterations=500, weight='weight')
    del pos['anchor']
    xy = np.array(list(pos.values()))
    xy = xy - (xy.max(axis=0) + xy.min(axis=0)) / 2
    xy = xy / np.linalg.norm(xy, axis=1).max() * 0.86
    pos = dict(zip(pos.keys(), xy))
    for i, n in enumerate(isolates):
        theta = 2 * np.pi * i / len(isolates) + np.pi / 4
        pos[n] = np.array([np.cos(theta), np.sin(theta)]) * 1.0
    return pos


def draw(g, pos, path):
    # Drawn near its size on the slide (1.2-1.6 in), so marker and line sizes read as they will
    fig, ax = plt.subplots(figsize=(1.6, 1.6), dpi=600)
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color=COLOR_EDGE, width=1.1, alpha=0.85)
    for sex, colour, marker in (('f', COLOR_F, 'o'), ('m', COLOR_M, 's')):
        nodes = [n for n, d in g.nodes(data=True) if d['sex'] == sex]
        sizes = [9 + 8 * g.degree(n) for n in nodes]
        nx.draw_networkx_nodes(g, pos, nodelist=nodes, node_color=colour, node_shape=marker,
                               node_size=sizes, linewidths=0.4, edgecolors='white', ax=ax)
    ax.set_xlim(-1.06, 1.06)
    ax.set_ylim(-1.06, 1.06)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path, transparent=True)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    for name, propensity, seed in NETWORKS:
        g = sample_network(propensity, seed)
        deg = np.array([d for _, d in g.degree()])
        print(f'{name:9s} edges={g.number_of_edges()} mean={deg.mean():.2f} '
              f'max={deg.max()} no-partner={np.mean(deg == 0):.0%}')
        draw(g, layout(g, seed), OUT / f'net_{name}.png')


if __name__ == '__main__':
    main()
