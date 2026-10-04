"""
export_natsal_figures.py
========================

Saves the Natsal-3 figures drawn in natsal_analysis_working.ipynb as print-resolution PNGs in
figs/Natsal/, for the paper. The notebook only keeps its plots as ~100 dpi inline outputs.

The notebook is not modified or re-saved. Its code cells are executed here, top to bottom, in one
namespace (IPython magics dropped, display() printed, plt.show() a no-op), and the figures a cell
leaves open are saved under the names in FIGURES. A cell that raises is reported and skipped, as
the notebook itself has a few cells that no longer run top to bottom.

FIGURES is keyed by code-cell index (0-based, counting all cells), so re-check it if cells are
added or moved in the notebook.

    python export_natsal_figures.py
"""
import json
import pathlib
import re
import traceback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parent
NOTEBOOK = REPO / 'natsal_analysis_working.ipynb'
OUT_DIR = REPO / 'figs' / 'Natsal'
DPI = 200

# (cell index, n-th figure the cell leaves open) -> output name
FIGURES = {
    (12, 0): 'natsal_age_mixing_cohabiting',
    (18, 0): 'natsal_age_mixing_input_matrices',
    (21, 0): 'natsal_debut_female_fits',
    (22, 0): 'natsal_debut_male_fits',
    (31, 0): 'natsal_annual_partners_fits',
    (33, 0): 'natsal_concurrency_by_age',
    (34, 0): 'natsal_partner_counts_by_window',
    (38, 0): 'natsal_concurrent_partners_point_female',
    (38, 1): 'natsal_concurrent_partners_point_male',
    (39, 0): 'natsal_max_concurrent_last_year_female',
    (39, 1): 'natsal_max_concurrent_last_year_male',
    (42, 0): 'natsal_durations_empirical',
    (43, 0): 'natsal_durations_exponential_fit',
    (47, 0): 'natsal_paid_partners',
}

MAGIC = re.compile(r'^\s*[%!]')


def main():
    cells = json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.show = lambda *a, **k: None
    ns = {'__name__': '__natsal__', 'display': print}
    saved, failed = [], []
    for i, cell in enumerate(cells):
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        src = '\n'.join(line for line in source.splitlines() if not MAGIC.match(line))
        plt.close('all')
        try:
            exec(compile(src, f'<cell {i}>', 'exec'), ns)
        except Exception:
            failed.append(i)
            print(f'cell {i} raised:\n{traceback.format_exc(limit=1)}')
        for k, num in enumerate(plt.get_fignums()):
            name = FIGURES.get((i, k))
            if name:
                out = OUT_DIR / f'{name}.png'
                plt.figure(num).savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
                saved.append(name)
                print(f'saved {out.relative_to(REPO)}')
    plt.close('all')
    missing = sorted(set(FIGURES.values()) - set(saved))
    print(f'\n{len(saved)} saved; cells that raised: {failed or "none"}; not produced: {missing or "none"}')


if __name__ == '__main__':
    main()
