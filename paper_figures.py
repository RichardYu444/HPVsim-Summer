"""
paper_figures.py
================

Builds Paper_Writeup/figures/ (the Overleaf project's figure folder) from the analysis figures in
figs/ and network_plots/, so the paper always shows the current version of each figure. Re-run it
after regenerating any figure below, then compile and push the paper.

    python paper_figures.py

Every entry in FIGURES is (source, destination[, crop]). Sources are relative to this folder;
destinations are relative to Paper_Writeup/figures/ and are what the .tex files reference.

crop picks one panel (or row of panels) out of a multi-panel diagnostic figure, so that single
panels can sit beside their Natsal-3 counterparts in the methods:

    ('dist', col, row)  one panel of the 4 x 2 network-distribution figure drawn by
                        run_r0_corrected_1900.plot_distributions / default_network_testing.py /
                        community_powerlaw_testing.py (figsize 21 x 10, tight_layout top 0.96).
                        Panels: row 0 = 1 instantaneous, 2 quarterly, 3 annual degree,
                        4 instantaneous by type; row 1 = 5 durations, 6 mean annual degree,
                        7 standing long fraction, 8 annual degree by type.
    ('mix', row)        one row of the 2 x 2 mixing figure (run_r0_corrected_1900.plot_mixing,
                        community_powerlaw_testing.py; tight_layout top 0.95). Row 0 = input and
                        realised age mixing, row 1 = degree by age.

Crops are trimmed to their content. A missing source is reported and skipped, not fatal.
"""
import pathlib
import shutil

from PIL import Image, ImageChops

REPO = pathlib.Path(__file__).resolve().parent
OUT = REPO / 'Paper_Writeup' / 'figures'

R0C = 'figs/R0 corrected'
G2_TAG = 'gamma2_meandeg1p4_vaxgeno_100k_1900_2050_50runs'
P35_TAG = 'powerlaw3p5_nogate_meandeg1p4_vaxgeno_100k_1900_2050_50runs'
G2_NET = f'{R0C}/gamma2_meandeg1p4/network/{G2_TAG}'
P35_NET = f'{R0C}/powerlaw_alpha3p5_nogate/network/{P35_TAG}'
DEF_NET = 'figs/Default/meandeg1p4/network/default_meandeg1p4'

# Network panels paired with Natsal in the methods: (name, dist-figure column, row)
DIST_PANELS = [('inst_degree', 0, 0), ('annual_degree', 2, 0), ('durations', 0, 1),
               ('mean_annual_degree', 1, 1)]
CURRENT_NETWORKS = {  # the mean-degree-1.4 networks used in results settings B and C
    'default': f'{DEF_NET}_network_distributions.png',
    'gamma2': f'{G2_NET}_network_distributions.png',
    'powerlaw3p5': f'{P35_NET}_network_distributions.png',
}

FIGURES = []

# ---- Methods: Natsal-3 (export_natsal_figures.py) ----
for name in ['annual_partners_fits', 'partner_counts_by_window', 'concurrency_by_age',
             'concurrent_partners_point_female', 'concurrent_partners_point_male',
             'max_concurrent_last_year_female', 'max_concurrent_last_year_male',
             'durations_exponential_fit', 'age_mixing_cohabiting', 'age_mixing_input_matrices',
             'debut_female_fits', 'debut_male_fits', 'paid_partners']:
    FIGURES.append((f'figs/Natsal/natsal_{name}.png', f'methods/natsal/{name}.png'))

# ---- Methods: current (mean degree 1.4) networks ----
for net, src in CURRENT_NETWORKS.items():
    FIGURES.append((src, f'methods/network/{net}_distributions.png'))
    for name, col, row in DIST_PANELS:
        FIGURES.append((src, f'methods/network/{net}_{name}.png', ('dist', col, row)))
for net, stem in [('gamma2', G2_NET), ('powerlaw3p5', P35_NET)]:
    FIGURES += [
        (f'{stem}_network_mixing.png', f'methods/network/{net}_mixing.png'),
        (f'{stem}_network_mixing.png', f'methods/network/{net}_age_mixing.png', ('mix', 0)),
        (f'{stem}_network_mixing.png', f'methods/network/{net}_degree_by_age.png', ('mix', 1)),
        (f'{stem}_network_graph_n1000.png', f'methods/network/{net}_graph_n1000.png'),
    ]
FIGURES.append(('network_plots/network_default_n1000.png', 'methods/network/default_graph_n1000.png'))

# ---- Methods: earlier network configurations (results setting A) ----
FIGURES += [
    ('figs/Network/default_network_testing_distributions2.png', 'methods/network_earlier/default_builtin_distributions.png'),
    ('figs/Network/community_powerlaw_testing_distributions.png', 'methods/network_earlier/powerlaw2p05_distributions.png'),
    ('figs/Power Law/powerlaw_log_log_testing_distributions.png', 'methods/network_earlier/powerlaw2p05_distributions_loglog.png'),
    ('figs/community/community_powerlaw_network_stats.png', 'methods/network_earlier/powerlaw2p05_calibrated_run.png'),
    ('figs/community/community_powerlaw_testing_mixing.png', 'methods/network_earlier/powerlaw2p05_age_mixing.png', ('mix', 0)),
    ('network_plots/network_grid_n1000.png', 'methods/network_earlier/network_grid_n1000.png'),
]
for shape in ['0p05', '0p25', '1p0', '2p5', '5p0']:
    FIGURES.append((f'figs/Network/network_stats_gamma{shape}_distributions.png',
                    f'methods/network_earlier/gamma{shape}_distributions.png'))

# ---- Results A: first network comparison (1980-2055, usual interventions) ----
A = 'results/A_initial'
for out, dflt, g3, pl in [
    ('hpv_prevalence', 'default_hpv_prev', 'gammashape3v1inter_hpv_prevalence', 'powerlawinter_hpv_prevalence'),
    ('infections', 'default_infections', 'gammashape3v1inter_infections', 'powerlawinter_infections'),
    ('cancer_incidence', 'default_cancer_inc', 'gammashape3v1inter_cancer_incidence', 'powerlawinter_cancer_incidence'),
]:
    FIGURES += [
        (f'figs/Default/{dflt}.png', f'{A}/default_{out}.png'),
        (f'figs/GammaSweep/{g3}.png', f'{A}/gamma3_{out}.png'),
        (f'figs/Power Law/{pl}.png', f'{A}/powerlaw2p05_{out}.png'),
    ]
FIGURES += [
    ('figs/Summary/graph6_networks_timeseries.png', f'{A}/networks_cancer_incidence_timeseries.png'),
    ('figs/Summary/graph1_networks_interventions.png', f'{A}/networks_cancer_incidence_2040.png'),
    ('figs/Summary/graph2_networks_no_interventions.png', f'{A}/networks_cancer_incidence_2040_no_interventions.png'),
    ('figs/Summary/graph5_gamma_sweep.png', f'{A}/gamma_sweep_cancer_incidence_2040.png'),
    ('figs/GammaSweep/compare_hpv_prevalence.png', f'{A}/gamma_sweep_hpv_prevalence.png'),
    ('figs/GammaSweep/compare_cancer_incidence.png', f'{A}/gamma_sweep_cancer_incidence.png'),
]

# ---- Results B: mean degree 1.4, calibrated genotypes (1950-2070) ----
B = 'results/B_meandeg1p4'
for net, stem in [
    ('default', 'figs/Default/meandeg1p4/default_meandeg1p4_100k_1950_2070_50runs'),
    ('gamma2', 'figs/Gamma/gamma2_nocomm/gamma2_nocomm_100k_1900_2050_50runs'),
    ('powerlaw3p5', 'figs/Power Law/alpha3p5_nogate/powerlaw_alpha3p5_nogate_100k_1950_2070_50runs'),
]:
    for out in ['hpv_prevalence', 'hpv_prevalence_by_genotype', 'infections', 'cancer_incidence',
                'cancer_incidence_by_genotype']:
        FIGURES.append((f'{stem}_{out}.png', f'{B}/{net}_{out}.png'))

# ---- Results C: R0-corrected (R0 1-1.5 per genotype, 1900-2050) ----
C = 'results/C_r0_corrected'
for net, stem in [('gamma2', f'{R0C}/gamma2_meandeg1p4/{G2_TAG}'),
                  ('powerlaw3p5', f'{R0C}/powerlaw_alpha3p5_nogate/{P35_TAG}')]:
    for out in ['acts_R0', 'prevalence_timelines', 'hpv_prevalence_by_genotype',
                'hpv_incidence_by_genotype', 'cin_prevalence_by_genotype', 'cancer_incidence',
                'cancer_incidence_by_genotype', 'cancers_by_genotype']:
        FIGURES.append((f'{stem}_{out}.png', f'{C}/{net}_{out}.png'))

# ---- Results: single-network analyses (no default/gamma/power-law set) ----
O = 'results/other'
FIGURES += [
    ('figs/Summary/graph3_core_group.png', f'{O}/core_group_cancer_incidence_2040.png'),
    ('figs/Summary/graph3b_core_group_paired.png', f'{O}/core_group_paired_change.png'),
    ('figs/Power Law/core group/core_vacc_alpha_compare_cancer_incidence.png', f'{O}/core_group_alpha_compare_cancer_incidence.png'),
    ('figs/Power Law/core group/powerlaw_alpha3_200k_50runs_lorenz.png', f'{O}/attribution_lorenz.png'),
    ('figs/Power Law/core group/powerlaw_alpha3_200k_50runs_attribution_curve.png', f'{O}/attribution_curve.png'),
    ('figs/Default/reff/default_20k_1950_2090_reff_by_genotype.png', f'{O}/default_reff_by_genotype.png'),
    ('figs/Default/reff/default_20k_1950_2090_r0_and_rvx.png', f'{O}/default_r0_and_rvx.png'),
    ('figs/Default/ohr_replacement/ohr_rise_mechanism.png', f'{O}/default_ohr_rise_mechanism.png'),
]


def trim(im, pad=12):
    """Crop to the non-background content (background = top-left pixel colour), plus a margin."""
    rgb = im.convert('RGB')
    bg = Image.new('RGB', rgb.size, rgb.getpixel((0, 0)))
    diff = ImageChops.difference(rgb, bg).convert('L').point(lambda v: 255 if v > 24 else 0)
    box = diff.getbbox()
    if box is None:
        return rgb
    l, t, r, b = box
    return rgb.crop((max(l - pad, 0), max(t - pad, 0), min(r + pad, rgb.width), min(b + pad, rgb.height)))


def row_gap(im, x0, x1, y0, y1):
    """Middle of the longest all-background horizontal run in im[y0:y1, x0:x1] (the gutter
    between two rows of panels), so a crop never cuts through an axis label or panel title."""
    band = im.convert('RGB').crop((x0, y0, x1, y1))
    bg = Image.new('RGB', band.size, im.convert('RGB').getpixel((0, 0)))
    ink = ImageChops.difference(band, bg).convert('L').point(lambda v: 255 if v > 24 else 0)
    rows = [ink.crop((0, y, band.width, y + 1)).getbbox() is None for y in range(band.height)]
    best, start = (0, 0), None
    for y, empty in enumerate(rows + [False]):
        if empty and start is None:
            start = y
        elif not empty and start is not None:
            best = max(best, (y - start, start))
            start = None
    length, start = best
    return y0 + start + length // 2 if length else (y0 + y1) // 2


def crop(im, spec):
    w, h = im.size
    if spec[0] == 'dist':
        _, col, row = spec
        x0, x1 = round(col * w / 4), round((col + 1) * w / 4)
        top = round(0.04 * h)
        mid = row_gap(im, x0, x1, round(0.45 * h), round(0.62 * h))
        box = (x0, top, x1, mid) if row == 0 else (x0, mid, x1, h)
    elif spec[0] == 'mix':
        _, row = spec
        top = round(0.05 * h)
        mid = row_gap(im, 0, w, round(0.45 * h), round(0.62 * h))
        box = (0, top, w, mid) if row == 0 else (0, mid, w, h)
    else:
        raise ValueError(f'unknown crop {spec!r}')
    return trim(im.crop(box))


def main():
    done, missing = 0, []
    for entry in FIGURES:
        src, dst = REPO / entry[0], OUT / entry[1]
        if not src.exists():
            missing.append(entry[0])
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if len(entry) == 3:
            with Image.open(src) as im:
                crop(im, entry[2]).save(dst, optimize=True)
        else:
            shutil.copyfile(src, dst)
        done += 1
    print(f'{done} figures written to {OUT.relative_to(REPO)}')
    for m in sorted(set(missing)):
        print(f'  missing source, skipped: {m}')


if __name__ == '__main__':
    main()
