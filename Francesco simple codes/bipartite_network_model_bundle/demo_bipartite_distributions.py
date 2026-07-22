#!/usr/bin/env python3
"""
demo_bipartite_distributions_revised
====================================

Calibrate a temporal bipartite network, run a stationary measurement period,
and plot eight diagnostics.

The early- and late-year curves provide a visual stationarity check. Degree is
reported at three observation scales:

* instantaneous degree: current partners in one monthly state;
* quarterly degree: distinct partners in a three-month union;
* annual degree: distinct partners in a twelve-month union.

Two additional panels split instantaneous and annual degree by partnership type.
For the annual type-specific counts, a partner is counted for a type when at
least one partnership episode of that type occurred during the year. A dyad can
therefore appear in both type-specific counts if it changed type during the year.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import simple_bipartite_network_model as M


N_U = 100_000
N_V = 100_000
YEARS = 21
EARLY_YEAR = 6
LATE_YEAR = 20
SEED = 0
OUT_PNG = "bipartite_distributions_revised2.png"

USER_PARAMS = {
    "mean_partners_per_year": 3.0,
    "gamma_shape": 1.5,
    "D_mean_short": 2.0,
    "D_mean_long": 36.0,
    "pi_long": 0.5,
    "tau": 360.0,
}

SHORT, LONG = M.EDGE_SHORT, M.EDGE_LONG
TYPE_LABEL = {SHORT: "short", LONG: "long"}
TYPE_COLOR = {SHORT: "C0", LONG: "C3"}
WINDOW_STYLE = {"early": "-", "late": "--"}


def degrees_from_edges(state, edge_mask=None):
    """Return U and V degree arrays for all edges or a selected edge subset."""
    degree_u = np.zeros(state["u_uid"].size, dtype=np.int64)
    degree_v = np.zeros(state["v_uid"].size, dtype=np.int64)
    if not state["edges_u"].size:
        return degree_u, degree_v

    edge_u = state["edges_u"]
    edge_v = state["edges_v"]
    if edge_mask is not None:
        edge_u = edge_u[edge_mask]
        edge_v = edge_v[edge_mask]
    if edge_u.size:
        np.add.at(degree_u, np.searchsorted(state["u_uid"], edge_u), 1)
        np.add.at(degree_v, np.searchsorted(state["v_uid"], edge_v), 1)
    return degree_u, degree_v


def integrated_degrees(state, pairs):
    """Return degree in a pair union, restricted to nodes active at its end."""
    degree_u = np.zeros(state["u_uid"].size, dtype=np.int64)
    degree_v = np.zeros(state["v_uid"].size, dtype=np.int64)
    if not pairs:
        return degree_u, degree_v

    pair_array = np.asarray(sorted(pairs), dtype=np.int64)
    edge_u, edge_v = pair_array[:, 0], pair_array[:, 1]
    alive = np.isin(edge_u, state["u_uid"]) & np.isin(edge_v, state["v_uid"])
    edge_u, edge_v = edge_u[alive], edge_v[alive]
    if edge_u.size:
        np.add.at(degree_u, np.searchsorted(state["u_uid"], edge_u), 1)
        np.add.at(degree_v, np.searchsorted(state["v_uid"], edge_v), 1)
    return degree_u, degree_v


def record_annual_partners(state, all_u, all_v, typed_u, typed_v):
    """Add the current edges to annual all-type and type-specific partner sets."""
    for u, v, edge_type in zip(
        state["edges_u"].tolist(),
        state["edges_v"].tolist(),
        state["edges_type"].tolist(),
    ):
        all_u.setdefault(u, set()).add(v)
        all_v.setdefault(v, set()).add(u)
        typed_u[edge_type].setdefault(u, set()).add(v)
        typed_v[edge_type].setdefault(v, set()).add(u)


def partner_counts(partners_u, partners_v, observed_u, observed_v):
    """Return pooled U and V distinct-partner counts for one observation window."""
    return np.asarray(
        [len(partners_u.get(uid, ())) for uid in observed_u]
        + [len(partners_v.get(uid, ())) for uid in observed_v],
        dtype=np.int64,
    )


def pmf(values, maximum):
    values = np.asarray(values, dtype=np.int64)
    histogram = np.bincount(values, minlength=maximum + 1)[: maximum + 1].astype(float)
    total = histogram.sum()
    return histogram / total if total else histogram


def selected_window(year):
    if year == EARLY_YEAR:
        return "early"
    if year == LATE_YEAR:
        return "late"
    return None


def safe_mean(values):
    values = np.asarray(values)
    return float(values.mean()) if values.size else float("nan")


def plot_pmf(ax, samples, labels, colors, styles, markers, xlabel, title):
    """Plot several discrete probability mass functions on a shared support."""
    maximum = max(int(np.max(sample)) for sample in samples if np.asarray(sample).size)
    x = np.arange(maximum + 1)
    for sample, label, color, style, marker in zip(
        samples, labels, colors, styles, markers
    ):
        ax.plot(
            x, pmf(sample, maximum), linestyle=style, marker=marker,
            markersize=3.5, color=color, alpha=0.85, label=label,
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("fraction of people")
    ax.set_yscale("log")
    ax.legend(fontsize=8)


def main():
    params = M.interpretable_to_params(USER_PARAMS, N_U, N_V)
    model = M.build_model(N_U, N_V)
    params = M.calibrate(model, params)
    rng = np.random.default_rng(SEED)
    state = M.init_network_state(model, params, rng)

    months_per_step = model["months_per_step"]
    burn = M.default_burn_months(params)
    target_k = USER_PARAMS["mean_partners_per_year"]
    target_f = params["frac_long_target"]
    p_form_long = params["p_form_long"]

    instantaneous_samples = {"early": [], "late": []}
    instantaneous_type_samples = {
        edge_type: {"early": [], "late": []} for edge_type in (SHORT, LONG)
    }
    integrated_samples = {"early": [], "late": []}
    yearly_samples = {"early": None, "late": None}
    yearly_type_samples = {
        edge_type: {"early": None, "late": None} for edge_type in (SHORT, LONG)
    }

    instantaneous_mean_t = []
    integrated_mean_t = []
    yearly_mean_t = []
    standing_long_t = []
    durations, duration_types = [], []

    partners_u, partners_v = {}, {}
    typed_partners_u = {SHORT: {}, LONG: {}}
    typed_partners_v = {SHORT: {}, LONG: {}}
    observed_u, observed_v = set(), set()
    quarter_pairs = set()

    for month in range(1, burn + 1):
        M.network_step(state, model, params, rng, month)
        edge_type = state["edges_type"]
        standing_long_t.append(
            float((edge_type == LONG).mean()) if edge_type.size else 0.0
        )

    for measurement_month in range(1, YEARS * 12 + 1):
        year = (measurement_month - 1) // 12 + 1
        window = selected_window(year)

        if (measurement_month - 1) % months_per_step == 0:
            quarter_pairs.update(zip(state["edges_u"].tolist(), state["edges_v"].tolist()))
        if (measurement_month - 1) % 12 == 0:
            observed_u.update(state["u_uid"].tolist())
            observed_v.update(state["v_uid"].tolist())
            record_annual_partners(
                state, partners_u, partners_v, typed_partners_u, typed_partners_v
            )

        M.network_step(
            state, model, params, rng, burn + measurement_month,
            track_durations=True,
            durations=durations,
            durations_type=duration_types,
        )

        degree_u, degree_v = degrees_from_edges(state)
        pooled = np.concatenate([degree_u, degree_v])
        instantaneous_mean_t.append(pooled.mean())
        if window:
            instantaneous_samples[window].append(pooled)

        for edge_type in (SHORT, LONG):
            mask = state["edges_type"] == edge_type
            type_u, type_v = degrees_from_edges(state, mask)
            if window:
                instantaneous_type_samples[edge_type][window].append(
                    np.concatenate([type_u, type_v])
                )

        edge_type = state["edges_type"]
        standing_long_t.append(
            float((edge_type == LONG).mean()) if edge_type.size else 0.0
        )

        observed_u.update(state["u_uid"].tolist())
        observed_v.update(state["v_uid"].tolist())
        record_annual_partners(
            state, partners_u, partners_v, typed_partners_u, typed_partners_v
        )
        quarter_pairs.update(zip(state["edges_u"].tolist(), state["edges_v"].tolist()))

        if measurement_month % months_per_step == 0:
            degree_u_q, degree_v_q = integrated_degrees(state, quarter_pairs)
            pooled_q = np.concatenate([degree_u_q, degree_v_q])
            integrated_mean_t.append(pooled_q.mean())
            if window:
                integrated_samples[window].append(pooled_q)
            quarter_pairs = set()

        if measurement_month % 12 == 0:
            counts = partner_counts(partners_u, partners_v, observed_u, observed_v)
            typed_counts = {
                edge_type: partner_counts(
                    typed_partners_u[edge_type], typed_partners_v[edge_type],
                    observed_u, observed_v,
                )
                for edge_type in (SHORT, LONG)
            }
            yearly_mean_t.append(counts.mean())
            if window:
                yearly_samples[window] = counts
                for edge_type in (SHORT, LONG):
                    yearly_type_samples[edge_type][window] = typed_counts[edge_type]

            partners_u, partners_v = {}, {}
            typed_partners_u = {SHORT: {}, LONG: {}}
            typed_partners_v = {SHORT: {}, LONG: {}}
            observed_u, observed_v = set(), set()

    durations_array = np.asarray(durations, dtype=float)
    duration_types_array = np.asarray(duration_types, dtype=np.int8)
    short_durations = durations_array[duration_types_array == SHORT]
    long_durations = durations_array[duration_types_array == LONG]

    instantaneous_pool = {
        label: np.concatenate(instantaneous_samples[label])
        for label in ("early", "late")
    }
    instantaneous_type_pool = {
        edge_type: {
            label: np.concatenate(instantaneous_type_samples[edge_type][label])
            for label in ("early", "late")
        }
        for edge_type in (SHORT, LONG)
    }
    integrated_pool = {
        label: np.concatenate(integrated_samples[label])
        for label in ("early", "late")
    }

    print("\n" + "=" * 68)
    print("CALIBRATED NETWORK SUMMARY")
    print("=" * 68)
    print(
        f"mean partners/year: realised {np.mean(yearly_mean_t):.3f}; "
        f"target {target_k:.3f}"
    )
    print(
        f"standing long fraction: realised {np.mean(standing_long_t[burn:]):.3f}; "
        f"target {target_f:.3f}; formation probability {p_form_long:.3f}"
    )
    interval_run = M.simulate(model, params, T=8, seed=1)
    interval_long = [
        float((matrix.data == 2).sum() / matrix.nnz)
        for matrix in interval_run["adj_type_list"] if matrix.nnz
    ]
    print(f"integrated quarterly long fraction: {np.mean(interval_long):.3f}")
    print(
        f"completed durations: short {safe_mean(short_durations):.2f} months; "
        f"long {safe_mean(long_durations):.2f} months"
    )
    print(
        "completed durations include dissolution and endpoint mortality; "
        "partnerships active at the end are right-censored"
    )
    print(
        f"mean degree: instantaneous {np.mean(instantaneous_mean_t):.3f}; "
        f"quarterly union {np.mean(integrated_mean_t):.3f}"
    )

    window_color = {"early": "C0", "late": "C3"}
    window_label = {"early": f"year {EARLY_YEAR}", "late": f"year {LATE_YEAR}"}

    fig, axes = plt.subplots(2, 4, figsize=(21, 10))

    plot_pmf(
        axes[0, 0],
        [instantaneous_pool["early"], instantaneous_pool["late"]],
        [window_label["early"], window_label["late"]],
        [window_color["early"], window_color["late"]],
        ["-", "-"], ["o", "o"],
        "current partners",
        "1. Instantaneous degree\n(monthly states pooled within year)",
    )

    plot_pmf(
        axes[0, 1],
        [integrated_pool["early"], integrated_pool["late"]],
        [window_label["early"], window_label["late"]],
        [window_color["early"], window_color["late"]],
        ["-", "-"], ["s", "s"],
        "partners in the quarter",
        f"2. Quarterly degree\n({months_per_step}-month union)",
    )

    ax = axes[0, 2]
    maximum = int(max(yearly_samples["early"].max(), yearly_samples["late"].max()))
    bins = np.arange(maximum + 2) - 0.5
    for label in ("early", "late"):
        ax.hist(
            yearly_samples[label], bins=bins, density=True, histtype="step",
            linewidth=2, color=window_color[label], label=window_label[label],
        )
    ax.axvline(target_k, linestyle="--", color="gray", label="target mean")
    ax.set_title("3. Annual degree\n(distinct partners in 12 months)")
    ax.set_xlabel("distinct partners")
    ax.set_ylabel("fraction of people")
    ax.set_yscale("log")
    ax.legend(fontsize=8)

    type_inst_samples = []
    type_inst_labels = []
    type_inst_colors = []
    type_inst_styles = []
    type_inst_markers = []
    for edge_type in (SHORT, LONG):
        for label in ("early", "late"):
            type_inst_samples.append(instantaneous_type_pool[edge_type][label])
            type_inst_labels.append(f"{TYPE_LABEL[edge_type]}, {window_label[label]}")
            type_inst_colors.append(TYPE_COLOR[edge_type])
            type_inst_styles.append(WINDOW_STYLE[label])
            type_inst_markers.append("o" if edge_type == SHORT else "s")
    plot_pmf(
        axes[0, 3], type_inst_samples, type_inst_labels,
        type_inst_colors, type_inst_styles, type_inst_markers,
        "current partners of the selected type",
        "4. Instantaneous degree by type",
    )

    ax = axes[1, 0]
    maximum = int(durations_array.max())
    bins = np.arange(maximum + 2) - 0.5
    ax.hist(short_durations, bins=bins, density=True, histtype="step",
            linewidth=2, color=TYPE_COLOR[SHORT], label="short")
    ax.hist(long_durations, bins=bins, density=True, histtype="step",
            linewidth=2, color=TYPE_COLOR[LONG], label="long")
    ax.axvline(USER_PARAMS["D_mean_short"], linestyle="--",
               color=TYPE_COLOR[SHORT], alpha=0.6)
    ax.axvline(USER_PARAMS["D_mean_long"], linestyle="--",
               color=TYPE_COLOR[LONG], alpha=0.6)
    ax.set_title("5. Completed partnership durations\n(endpoint mortality included)")
    ax.set_xlabel("duration in months")
    ax.set_ylabel("fraction of completed partnerships")
    ax.set_yscale("log")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    years = np.arange(1, YEARS + 1)
    ax.axhline(target_k, linestyle="--", color="gray", label="target")
    ax.plot(years, yearly_mean_t, "o-", color="C2", label="realised")
    ax.set_ylim(0, max(max(yearly_mean_t), target_k) * 1.3)
    ax.set_title("6. Mean annual degree")
    ax.set_xlabel("measurement year")
    ax.set_ylabel("mean distinct partners")
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    months = np.arange(1, len(standing_long_t) + 1)
    ax.plot(months, standing_long_t, color=TYPE_COLOR[LONG], linewidth=1.2,
            label="standing fraction")
    ax.axhline(target_f, linestyle="--", color="gray", label="target")
    ax.axhline(p_form_long, linestyle=":", color=TYPE_COLOR[SHORT],
               label=f"formation probability = {p_form_long:.2f}")
    ax.axvline(burn, color="k", linewidth=0.8, alpha=0.4)
    ax.text(burn, 0.02, " end of burn-in", fontsize=8, alpha=0.6)
    ax.set_ylim(0, 1)
    ax.set_title("7. Standing long-partnership fraction")
    ax.set_xlabel("month, including burn-in")
    ax.set_ylabel("fraction of active edges")
    ax.legend(loc="center right", fontsize=8)

    annual_type_samples = []
    annual_type_labels = []
    annual_type_colors = []
    annual_type_styles = []
    annual_type_markers = []
    for edge_type in (SHORT, LONG):
        for label in ("early", "late"):
            annual_type_samples.append(yearly_type_samples[edge_type][label])
            annual_type_labels.append(f"{TYPE_LABEL[edge_type]}, {window_label[label]}")
            annual_type_colors.append(TYPE_COLOR[edge_type])
            annual_type_styles.append(WINDOW_STYLE[label])
            annual_type_markers.append("o" if edge_type == SHORT else "s")
    plot_pmf(
        axes[1, 3], annual_type_samples, annual_type_labels,
        annual_type_colors, annual_type_styles, annual_type_markers,
        "distinct partners of the selected type",
        "8. Annual degree by type\n(12-month union)",
    )

    fig.suptitle(
        f"Bipartite temporal network — N={N_U}+{N_V}, partners/year={target_k}, "
        f"Gamma shape={USER_PARAMS['gamma_shape']}, durations="
        f"{USER_PARAMS['D_mean_short']}/{USER_PARAMS['D_mean_long']} months, "
        f"standing long fraction={target_f}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_PNG, dpi=120)
    print(f"\nsaved figure -> {OUT_PNG}")


if __name__ == "__main__":
    main()
