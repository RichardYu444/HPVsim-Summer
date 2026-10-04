import pandas as pd
import numpy as np
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import matplotlib.pyplot as plt


FILES = {
    "Random": "RandomAllParam.csv",
    "Default": "defaultallparam02Feb.csv",
    "Core Group": "CoreGroupAllParam.csv",
}

SEED_COL = "Seed"
YEAR_COL = "year"

COLUMN_ALIASES = {
    "hpv_prev": ["hpv_prevalence"],
    "cancers": ["cancer_incidence"],
}

def z_normalise(series: np.ndarray) -> np.ndarray:
    std = np.std(series)
    if std == 0:
        return series - np.mean(series)  # avoid divide-by-zero
    return (series - np.mean(series)) / std


def find_column(df: pd.DataFrame, aliases: list[str], label: str) -> str:
    for col in aliases:
        if col in df.columns:
            return col
    raise KeyError(
        f"Could not find a column for '{label}'. Tried: {aliases}. "
        f"Available columns include: {df.columns.tolist()}"
    )


def prepare_median_timeseries(path: str, logical_col: str) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(path)
    value_col = find_column(df, COLUMN_ALIASES[logical_col], logical_col)

    cleaned = (
        df.groupby([SEED_COL, YEAR_COL], as_index=False)[value_col]
        .mean()
    )

    ts = (
        cleaned.groupby(YEAR_COL)[value_col]
        .median()
        .reset_index()
        .sort_values(YEAR_COL)
    )
    return ts, value_col

def plot_dtw_alignment(ts1, ts2, value_col, label1, label2):
    merged = ts1.merge(ts2, on=YEAR_COL, suffixes=("_1", "_2"))

    x = merged[f"{value_col}_1"].to_numpy()
    y = merged[f"{value_col}_2"].to_numpy()

    path = dtw.warping_path(x, y)

    plt.figure(figsize=(10, 6))

    # Plot both series
    plt.plot(x, label=label1)
    plt.plot(y, label=label2)

    # Draw warping lines
    for i, j in path:
        plt.plot([i, j], [x[i], y[j]], 'k-', alpha=0.3)

    plt.legend()
    plt.title(f"DTW Alignment: {label1} vs {label2}")
    plt.xlabel("Time index")
    plt.ylabel(value_col)
    #plt.show()



def compare_series_dtw(ts1: pd.DataFrame, ts2: pd.DataFrame, value_col: str) -> dict:
    merged = ts1.merge(ts2, on=YEAR_COL, suffixes=("_1", "_2"))

    x = merged[f"{value_col}_1"].to_numpy()
    y = merged[f"{value_col}_2"].to_numpy()

    # Compute DTW distance
    x = z_normalise(x)
    y = z_normalise(y)
    normalised_distance = dtw.distance(x, y)


    return {
        "n_years": len(merged),
        "dtw_normalised": normalised_distance,
    }


def run_metric(logical_col: str) -> pd.DataFrame:
    series = {}
    actual_column_name = None

    for label, path in FILES.items():
        ts, resolved_col = prepare_median_timeseries(path, logical_col)
        series[label] = ts
        actual_column_name = resolved_col

    labels = list(series.keys())
    results = []

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            stats = compare_series_dtw(
                series[labels[i]],
                series[labels[j]],
                actual_column_name
            )
            plot_dtw_alignment(
            series[labels[i]],
            series[labels[j]],
            actual_column_name,
            labels[i],
            labels[j]
            )
            stats["metric_requested"] = logical_col
            stats["csv_column_used"] = actual_column_name
            stats["comparison"] = f"{labels[i]} vs {labels[j]}"
            results.append(stats)

    return pd.DataFrame(results)[[
        "comparison",
        "dtw_normalised",
    ]]


def main() -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: f"{x:.6f}")

    print("\n=== DTW for HPV prevalence time series ===")
    hpv_results = run_metric("hpv_prev")
    print(hpv_results.to_string(index=False))

    print("\n=== DTW for cancer time series ===")
    cancer_results = run_metric("cancers")
    print(cancer_results.to_string(index=False))


if __name__ == "__main__":
    main()