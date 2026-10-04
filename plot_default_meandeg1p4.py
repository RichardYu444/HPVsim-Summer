"""
plot_default_meandeg1p4.py
==========================

The figures for run_default_meandeg1p4.py's 50 runs that plot_IQR.py does not draw: the two columns
that run added (prevalence*, infections*) and the per-genotype results. The standard outcomes come
from plot_IQR.py itself, into the same folder:

    python plot_IQR.py csvs/default_meandeg1p4_100k_1950_2070_50runs.csv default_meandeg1p4_100k_1950_2070_50runs "figs/Default/meandeg1p4" 2070 1950
    python plot_default_meandeg1p4.py

This is the default-network counterpart of plot_powerlaw_nogate.py, and it is deliberately not a
copy of it: it imports that module and repoints TAG/CSV/FIG_DIR, so both runs' figures are drawn by
literally the same code. That is the point -- the two CSVs are column-identical by construction
(run_default_meandeg1p4.py matches run_powerlaw_nogate.py's export exactly), and a figure meant to
be compared against another should not be drawn by a second, separately-drifting copy of the
plotting code. An edit to the palette, the IQR style or the end labels in plot_powerlaw_nogate.py
therefore lands on both sets at once.

Importing it is safe: its own main() is guarded by __name__ == '__main__', and its only import-time
effect is setting plot_IQR.XLIM_LEFT/RIGHT to 1950/2070 -- which is the x range this run wants too.

Writes figs/Default/meandeg1p4/<TAG>_<name>.png for each name below:

    prevalence_star                prevalence* over hpv_prevalence, and the gap between them
    infections_star                infections* -- people currently infected, NOT new infections
    <value>_by_genotype            hpv_prevalence, infections, cancer_incidence, one line per genotype
"""
import pathlib

import matplotlib
matplotlib.use('Agg')

import plot_powerlaw_nogate as figs  # also sets plot_IQR.XLIM_LEFT/RIGHT to 1950/2070

REPO = pathlib.Path(__file__).resolve().parent
TAG = 'default_meandeg1p4_100k_1950_2070_50runs'
CSV = REPO / 'csvs' / f'{TAG}.csv'
FIG_DIR = REPO / 'figs' / 'Default' / 'meandeg1p4'


def main():
    # figs.save() reads TAG and FIG_DIR as module globals and figs.main() reads CSV, so repointing
    # the three is the whole of the configuration -- everything else (palette, BY_GENOTYPE, the
    # axis styling) is network-agnostic and stays as it is.
    figs.TAG, figs.CSV, figs.FIG_DIR = TAG, CSV, FIG_DIR
    figs.main()


if __name__ == '__main__':
    main()
