import networkx as nx
import matplotlib.pyplot as plt
import hpvsim_working as hpv


def seed_layer_graphs(nh):
    """Create one nx.Graph per layer, seeded with the initial network snapshot."""
    graphs = {lkey: nx.Graph() for lkey in nh.layer_map}
    snap = nh.initial_snapshot

    for lkey, G in graphs.items():
        # Use the full alive population as the node set (not just people with a partner in this layer) so average degree means "average partners per living person",
        # and people with no partner in this layer correctly contribute a degree of 0.
        G.add_nodes_from(snap.added_nodes.uid.tolist())

    for eid, f, m, layer in zip(snap.added_edges.eid, snap.added_edges.f,
                                 snap.added_edges.m, snap.added_edges.layer):
        lkey = nh.layer_map[layer]
        graphs[lkey].add_edge(int(f), int(m), eid=int(eid))

    return graphs


def apply_delta(graphs, nh, delta):
    """Mutate each layer's graph in place to reflect one timestep's NetworkDelta."""
    added_nodes = delta.added_nodes.uid.tolist()
    removed_nodes = delta.removed_nodes.uid.tolist()

    # Group this timestep's edge changes by layer up front
    removed_by_layer = {lkey: [] for lkey in nh.layer_map}
    for f, m, layer in zip(delta.removed_edges.f, delta.removed_edges.m, delta.removed_edges.layer):
        removed_by_layer[nh.layer_map[layer]].append((int(f), int(m)))

    added_by_layer = {lkey: [] for lkey in nh.layer_map}
    for eid, f, m, layer in zip(delta.added_edges.eid, delta.added_edges.f,
                                 delta.added_edges.m, delta.added_edges.layer):
        added_by_layer[nh.layer_map[layer]].append((int(f), int(m), int(eid)))

    # Order matters here for two reasons that pull in opposite directions:
    #  - A partnership can dissolve and reform between the same two people within a single
    #    timestep (old edge removed, new edge with a *new* eid added). Edge removal must
    #    happen before the new edge is added, or has_edge(f, m) would match -- and delete --
    #    the fresh edge instead of the stale one.
    #  - A node can be added and removed within the same timestep (e.g. a transient entry).
    #    Node removal must happen *last*, after any edges involving it have been added, so it
    #    (and its incident edges, dropped automatically by networkx) end up gone at the end.
    for lkey, G in graphs.items():
        # 1) Remove edges (endpoints are still guaranteed present at this point)
        for f, m in removed_by_layer[lkey]:
            if G.has_edge(f, m):
                G.remove_edge(f, m)

        # 2) Add new nodes
        G.add_nodes_from(added_nodes)

        # 3) Add new edges (safe even for a same-pair reform, since step 1 already cleared it)
        for f, m, eid in added_by_layer[lkey]:
            G.add_edge(f, m, eid=eid)

        # 4) Remove nodes last (silently ignores any already gone; drops incident edges too)
        G.remove_nodes_from(removed_nodes)

    return


def avg_degree(G):
    n = G.number_of_nodes()
    if n == 0:
        return 0.0
    return 2 * G.number_of_edges() / n


def cross_check(graphs, nh, t):
    """Independently rebuild each layer's graph from scratch at timestep t and compare
    it against the incrementally-built graph, as a correctness check on the delta stream."""
    nodes = nh.nodes_at(t)
    edges = nh.edges_at(t)

    for lkey, G in graphs.items():
        ref = nx.Graph()
        ref.add_nodes_from(nodes)
        for f, m, lk in edges.values():
            if lk == lkey:
                ref.add_edge(f, m)

        assert set(G.nodes()) == set(ref.nodes()), f'Node mismatch in layer "{lkey}" at t={t}'
        assert set(map(frozenset, G.edges())) == set(map(frozenset, ref.edges())), \
            f'Edge mismatch in layer "{lkey}" at t={t}'

    return


def main():
    sim = hpv.Sim(n_agents=2000, start=2000, n_years=20, dt=0.25, network='default',
                  verbose=0, rand_seed=1, analyzers=[hpv.network_history()])
    sim.run()

    nh = sim.get_analyzer('network_history')
    graphs = seed_layer_graphs(nh)

    years = [sim.yearvec[0]]
    series = {lkey: [avg_degree(graphs[lkey])] for lkey in nh.layer_map}

    for t in sorted(nh.deltas.keys()):
        apply_delta(graphs, nh, nh.deltas[t])
        years.append(sim.yearvec[t + 1] if t + 1 < len(sim.yearvec) else sim.yearvec[t] + sim['dt'])
        for lkey in nh.layer_map:
            series[lkey].append(avg_degree(graphs[lkey]))

    # Correctness check: incremental replay must match an independent from-scratch
    # reconstruction at the final timestep.
    t_last = sim.npts - 1
    cross_check(graphs, nh, t_last)
    print(f'Cross-check passed: incremental graphs match nh.nodes_at/edges_at at t={t_last}')

    print('\nAverage degree by layer:')
    header = 'year'.ljust(8) + ''.join(lkey.ljust(10) for lkey in nh.layer_map)
    print(header)
    for i, year in enumerate(years):
        row = f'{year:<8.1f}' + ''.join(f'{series[lkey][i]:<10.4f}' for lkey in nh.layer_map)
        print(row)

    fig, ax = plt.subplots(figsize=(8, 5))
    for lkey in nh.layer_map:
        ax.plot(years, series[lkey], marker='o', label=f'layer "{lkey}"')
    ax.set_xlabel('Year')
    ax.set_ylabel('Average degree')
    ax.set_title('Average degree over time, by network layer')
    ax.legend()
    fig.tight_layout()
    out_path = 'network_degree_timeseries.png'
    fig.savefig(out_path)
    print(f'\nPlot saved to {out_path}')

    return years, series


if __name__ == '__main__':
    main()
