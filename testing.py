import hpvsim_working as hpv
import numpy as np

sim = hpv.Sim(n_agents=500, start=2000, n_years=5, dt=1.0, network='default', verbose=0,
              analyzers=[hpv.network_history()])
sim.run()

nh = sim.get_analyzer('network_history')
print(nh)
t_last = sim.npts - 1
nodes_end = nh.nodes_at(t_last)
edges_end = nh.edges_at(t_last)

people = sim.people
raw_alive = int(people.alive.sum())
print('raw alive agents:', raw_alive, 'reconstructed:', len(nodes_end))
assert raw_alive == len(nodes_end), 'MISMATCH nodes'

actual_edge_count = sum(len(layer['f']) for layer in people.contacts.values())
print('raw contact edges:', actual_edge_count, 'reconstructed edges:', len(edges_end))
assert actual_edge_count == len(edges_end), 'MISMATCH edges'

for lkey in nh.layer_map:
    actual_pairs = set(zip(people.contacts[lkey]['f'].tolist(), people.contacts[lkey]['m'].tolist()))
    recon_pairs = set((f,m) for (f,m,lk) in edges_end.values() if lk==lkey)
    print('layer', lkey, 'actual pairs:', len(actual_pairs), 'recon pairs:', len(recon_pairs))
    assert actual_pairs == recon_pairs, 'MISMATCH pairs for layer '+lkey

print('ALL CHECKS PASSED')