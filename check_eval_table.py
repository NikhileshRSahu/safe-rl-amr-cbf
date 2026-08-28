import csv
import sys
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else "episode_summary.csv"

with open(path, newline="") as f:
    rows = list(csv.DictReader(f))

n = len(rows)
if n == 0:
    print("No rows found -- is this the right file?")
    sys.exit(1)

# Pull the raw boolean/string columns straight off the CSV. csv.DictReader
# gives everything as strings, so parse the two truthy columns explicitly
# rather than trusting any derived label.
def as_bool(s):
    return str(s).strip().lower() in ("true", "1")

n_success = sum(as_bool(r["success"]) for r in rows)
n_collision = sum(as_bool(r["collision"]) for r in rows)
n_timeout = sum(
    (not as_bool(r["success"])) and (not as_bool(r["collision"]))
    for r in rows
)

collision_types = Counter(
    r["collision_type"] for r in rows if as_bool(r["collision"])
)

print(f"Rows read: {n}")
print()
print(f"Success Rate: {100*n_success/n:.0f}% ({n_success}/{n} episodes)")
print(f"Timeouts:     {100*n_timeout/n:.0f}% ({n_timeout}/{n} episodes)")
print(f"Collisions:   {100*n_collision/n:.0f}% ({n_collision}/{n} episodes)")
print()
print("Collision types:")
for ctype, count in collision_types.most_common():
    print(f"  {ctype}: {count}")

# Sanity check: every row should be exactly one of success/collision/timeout.
# If this fails, something in the CSV itself is inconsistent (e.g. a row
# marked both success and collision), independent of any percentage math.
overlap = sum(as_bool(r["success"]) and as_bool(r["collision"]) for r in rows)
accounted = n_success + n_collision + n_timeout
print()
if overlap:
    print(f"WARNING: {overlap} row(s) marked BOTH success and collision -- inconsistent raw data.")
if accounted != n:
    print(f"WARNING: counts don't add up to total rows ({accounted} != {n}) -- check the CSV by hand.")
else:
    print("Counts add up to total rows. No inconsistency found in the raw data.")
