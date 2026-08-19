"""Compare two NPZ files saved by dump_reset_state.py for exact equality."""
import argparse
from pathlib import Path
import numpy as np


def load_npz(p: Path):
    data = np.load(p, allow_pickle=True)
    return {k: data[k] for k in data.files}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a", type=str)
    p.add_argument("b", type=str)
    args = p.parse_args()
    pa = Path(args.a)
    pb = Path(args.b)
    da = load_npz(pa)
    db = load_npz(pb)

    keys = sorted(set(list(da.keys()) + list(db.keys())))
    identical = True
    for k in keys:
        va = da.get(k)
        vb = db.get(k)
        try:
            if not np.array_equal(va, vb):
                print(f"DIFFER: key={k}")
                print(f"  a: {va}")
                print(f"  b: {vb}")
                identical = False
            else:
                print(f"SAME: key={k}")
        except Exception as e:
            print(f"COMPARE ERROR key={k}: {e}")
            identical = False

    if identical:
        print("FILES IDENTICAL")
        return 0
    else:
        print("FILES DIFFER")
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
