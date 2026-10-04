"""
plot_gamma2_nocomm.py
=====================

The figures for run_gamma2_nocomm_1900.py's 50 runs, drawn by the same code as its two siblings
(plot_powerlaw_nogate.py, plot_default_meandeg1p4.py) and over the same 1950-2070 x range, so the
three sit side by side. The run itself ends in 2050, so the last 20 years of each panel are empty.
The standard outcomes come from plot_IQR.py, into the same folder:

    python plot_IQR.py csvs/gamma2_nocomm_100k_1900_2050_50runs.csv gamma2_nocomm_100k_1900_2050_50runs "figs/Gamma/gamma2_nocomm" 2070 1950
    python plot_gamma2_nocomm.py

Writes figs/Gamma/gamma2_nocomm/<TAG>_<name>.png for plot_powerlaw_nogate.py's names
(prevalence_star, infections_star, <value>_by_genotype).
"""
import pathlib

import matplotlib
matplotlib.use('Agg')

import plot_powerlaw_nogate as figs  # also sets plot_IQR.XLIM_LEFT/RIGHT to 1950/2070

REPO = pathlib.Path(__file__).resolve().parent
TAG = 'gamma2_nocomm_100k_1900_2050_50runs'
CSV = REPO / 'csvs' / f'{TAG}.csv'
FIG_DIR = REPO / 'figs' / 'Gamma' / 'gamma2_nocomm'


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figs.TAG, figs.CSV, figs.FIG_DIR = TAG, CSV, FIG_DIR
    figs.main()


if __name__ == '__main__':
    main()
