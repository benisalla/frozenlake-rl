"""06: every method on one environment, one table and three figures (~1 minute on the 4x4 lake).

`compare()` trains each method with its tuned defaults, scores them all the same way and writes
results.md (the comparison table), curves.png, policies.png and train_returns.png. Pass another
environment to compare there (slower: the slippery 8x8 lake takes about 15 minutes).

    python examples/06_compare_everything.py                      # frozenlake-4x4
    python examples/06_compare_everything.py frozenlake-4x4-slippery
    python examples/06_compare_everything.py taxi td               # DP + TD family + Dyna-Q
"""

import sys
from pathlib import Path

from frozenlake_rl.compare import compare, parse_methods

env = sys.argv[1] if len(sys.argv) > 1 else "frozenlake-4x4"
group = sys.argv[2] if len(sys.argv) > 2 else "all"
out = Path("examples/output/compare") / env
compare(env, parse_methods(group), out, eval_episodes=500)
print(f"\nwrote {out}/results.md, curves.png, policies.png, train_returns.png")
