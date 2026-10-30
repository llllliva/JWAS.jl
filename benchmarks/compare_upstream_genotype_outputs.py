"""Compare paired output exports: python script.py baseline_dir port_dir report_dir."""
import csv
import math
import sys
from pathlib import Path

baseline, port, report = map(Path, sys.argv[1:])
report.mkdir(parents=True, exist_ok=True)

def read_keyed(path, keys, expected):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    keyed = {}
    for row in rows:
        key = tuple(row[k].strip().strip('"') for k in keys)
        assert key not in keyed, (path, "duplicate", key)
        keyed[key] = row
    assert len(rows) == len(expected), (path, len(rows), len(expected))
    assert set(keyed) == expected, (path, "key-set mismatch")
    return keyed

summaries = []
for method in ("BayesC", "BayesR"):
    for seed in (73, 123):
        for block in ("false", "20"):
            prefix = f"{method}_seed{seed}_block{block}"
            summary = dict(method=method, seed=seed, block=block)
            for kind, keys, expected, columns in (
                ("animals", ("ID",), {(f"animal_{i:04d}",) for i in range(1, 129)}, ("EBV", "PEV")),
                ("markers", ("Trait", "Marker_ID"), {("y", f"rs_{1000+7*i}") for i in range(1, 258)}, ("Estimate", "SD", "Model_Frequency")),
            ):
                left = read_keyed(baseline / f"{prefix}_{kind}.csv", keys, expected)
                right = read_keyed(port / f"{prefix}_{kind}.csv", keys, expected)
                assert left.keys() == right.keys()
                aligned = []
                for key in sorted(expected):
                    row = dict(zip(keys, key))
                    for col in columns:
                        a, b = float(left[key][col]), float(right[key][col])
                        assert math.isfinite(a) and math.isfinite(b)
                        assert math.isclose(a, b, abs_tol=1e-4, rel_tol=1e-4), (prefix, key, col, a, b)
                        row[f"upstream_{col}"] = a
                        row[f"port_{col}"] = b
                    aligned.append(row)
                assert len(aligned) == len(expected)
                with (report / f"{prefix}_upstream_{kind}.csv").open("w") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(aligned[0]), lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(aligned)
                summary[f"{kind}_count"] = len(aligned)
                for col in columns:
                    summary[f"max_{col}_difference"] = max(abs(row[f"upstream_{col}"] - row[f"port_{col}"]) for row in aligned)
                if method == "BayesC" and seed == 73 and block == "false":
                    print(kind, "aligned sample:", aligned[:2])
            summaries.append(summary)
with (report / "upstream_output_comparison.csv").open("w") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summaries[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(summaries)
print("All eight upstream/port comparisons passed.")
