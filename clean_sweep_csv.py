import sys
import pathlib

# CSVs built by repeatedly calling df.to_csv(path, mode='a') (e.g. run_sim_gamma_sweep.py)
# end up with the header row repeated once per append. This strips every header
# occurrence after the first so the file has a single leading header, in place.


def clean_csv(path):
    path = pathlib.Path(path)
    lines = path.read_text().splitlines(keepends=True)
    if not lines:
        return
    header = lines[0]
    cleaned = [header] + [line for line in lines[1:] if line != header]
    path.write_text(''.join(cleaned))
    print(f'{path.name}: {len(lines)} -> {len(cleaned)} lines '
          f'(dropped {len(lines) - len(cleaned)} duplicate header rows)')


if __name__ == '__main__':
    targets = sys.argv[1:]
    if not targets:
        targets = [
            'gammashape_0p05.csv',
            'gammashape_0p25.csv',
            'gammashape_1p0.csv',
            'gammashape_2p5.csv',
            'gammashape_5p0.csv',
        ]
    for t in targets:
        clean_csv(t)
