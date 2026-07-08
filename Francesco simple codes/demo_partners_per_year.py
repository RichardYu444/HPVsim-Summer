#!/usr/bin/env python3
"""Simulate the toy network for 10 years and plot mean partners per year."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import simple_network_model as M

N, YEARS = 2000, 10
user_params = {"mean_partners_per_year": 3.0, "exponent": 2.3,
               "kappa": 60.0, "D_mean": 6.0}

params = M.interpretable_to_params(user_params, N, [N])
model  = M.build_model(N, [N], params)
rng    = np.random.default_rng(0)
state  = M.init_network_state(model, params, rng)

yearly, dist_first, dist_last = [], None, None
partners, active = {}, set()                        # distinct partners this year
for month in range(1, YEARS * 12 + 1):
    M.network_step(state, model, params, rng, month)
    active.update(state["u_uid"].tolist())          # everyone alive counts (incl. 0 partners)
    for a, b in zip(state["edges_u"].tolist(), state["edges_v"].tolist()):
        partners.setdefault(a, set()).add(b)
        partners.setdefault(b, set()).add(a)
    if month % 12 == 0:                             # close out the year
        counts = np.array([len(partners.get(u, ())) for u in active])
        yearly.append(counts.mean())
        if month == 12:        dist_first = counts
        if month == YEARS * 12: dist_last = counts
        partners, active = {}, set()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.axhline(user_params["mean_partners_per_year"], ls="--", color="gray", label="target input")
ax1.plot(range(1, YEARS + 1), yearly, "o-", color="C0", label="simulated")
ax1.set_xlabel("year"); ax1.set_ylabel("mean partners per year")
ax1.set_title("Mean partners per year"); ax1.set_ylim(0, max(yearly) * 1.3); ax1.legend()

bins = np.arange(0, max(dist_first.max(), dist_last.max()) + 2) - 0.5
ax2.hist(dist_first, bins=bins, alpha=0.6, label="year 1", color="C0")
ax2.hist(dist_last,  bins=bins, alpha=0.6, label="year 10", color="C3")
ax2.set_yscale("log"); ax2.set_xlabel("partners per year"); ax2.set_ylabel("nodes")
ax2.set_title("Distribution: start vs end"); ax2.legend()

fig.tight_layout()
fig.savefig("partners_per_year.png", dpi=130)
print("yearly means:", [round(v, 2) for v in yearly])
