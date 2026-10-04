"""
default_network_testing_meandeg1p4.py
=====================================

default_network_testing.py's eight-panel network diagnostic, but for the default network as it is
actually run in run_default_meandeg1p4.py (basePars.py's Natsal-calibrated partnership knobs,
mean degree 1.4 excluding singles) rather than HPVsim's built-in default pars.

Shaped to sit beside the Gamma-2 / power-law 3.5 diagnostics from run_r0_corrected_1900.py:
network-only (no interventions, no initial infection), 100,000 agents, 60 years from the run's
1950 start, early/late windows at years 3 and 50. default_network_testing.main() does all the
measuring and plotting; only the Sim it builds is swapped for one made from
run_default_meandeg1p4.make_pars().

    python default_network_testing_meandeg1p4.py

Writes figs/Default/meandeg1p4/network/default_meandeg1p4_network_distributions.png
"""
import pathlib

import default_network_testing as dnt
import run_default_meandeg1p4 as rdm

REPO = pathlib.Path(__file__).resolve().parent
OUT_DIR = REPO / 'figs' / 'Default' / 'meandeg1p4' / 'network'

dnt.N_AGENTS = 100_000
dnt.START = rdm.START
dnt.YEARS = 60
dnt.EARLY_YEAR, dnt.LATE_YEAR = 3, 50
dnt.SEED = 0
dnt.OUT_PNG = str(OUT_DIR / 'default_meandeg1p4_network_distributions.png')

_Sim = dnt.hpv.Sim


def _basepars_sim(**kwargs):
    pars = rdm.make_pars(
        dnt.SEED,
        n_agents=dnt.N_AGENTS,
        start=dnt.START,
        end=dnt.START + dnt.YEARS,
        interventions=[],
        rel_init_prev=0.0,
        analyzers=kwargs['analyzers'],
        verbose=0,
    )
    return _Sim(pars)


if __name__ == '__main__':
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dnt.hpv.Sim = _basepars_sim
    try:
        dnt.main()
    finally:
        dnt.hpv.Sim = _Sim
