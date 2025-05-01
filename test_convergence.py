import re
import pickle
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Process, Queue

import sys
sys.path.append("../")

from ABAF import ABAF
from semantics.bsafDiscreteModular import DiscreteModular

from semantics.modular.ProductAggregation        import ProductAggregation
from semantics.modular.SetProductAggregation     import SetProductAggregation
from semantics.modular.SumAggregation            import SumAggregation
from semantics.modular.LinearInfluence           import LinearInfluence
from semantics.modular.QuadraticMaximumInfluence import QuadraticMaximumInfluence

# ─── Config ─────────────────────────────────────────────────────────
INPUT_DIR       = Path("../dependency-graph-alternative/input_data_nf").resolve()
OUTPUT_PKL      = "convergence_results.pkl"
MAX_FILES       = 0       # 0 = no limit
MIN_SENTENCES   = 0
MAX_SENTENCES   = 25
TIMEOUT_SECONDS = 60      # per‐file timeout

# pick only .aba with s between MIN_SENTENCES and MAX_SENTENCES
pattern_s = re.compile(r"_s(\d+)_")
all_aba = sorted(INPUT_DIR.glob("*.aba"))
aba_paths = [
    p for p in all_aba
    if (m := pattern_s.search(p.name))
       and MIN_SENTENCES <= int(m.group(1)) <= MAX_SENTENCES
]
if MAX_FILES > 0:
    aba_paths = aba_paths[:MAX_FILES]

# extract the parameters from filename
param_pat = re.compile(
    r"_s(?P<s>\d+)_"
    r"c(?P<c>[\d.]+)_"
    r"n(?P<n>[\d.]+)_"
    r"a(?P<a>[\d.]+)_"
    r"r(?P<r>\d+)_"
    r"b(?P<b>\d+)"
)

# the two models we want to run, per file
RUNS = [
    ("DF-QuAD", dict(
        aggregation    = ProductAggregation(),
        influence      = LinearInfluence(conservativeness=1),
        set_aggregation= SetProductAggregation()
    )),
    ("QE", dict(
        aggregation    = SumAggregation(),
        influence      = QuadraticMaximumInfluence(conservativeness=1),
        set_aggregation= SetProductAggregation()
    ))
]


def worker_file(aba_path_str, params, runs, queue):
    """
    Load ABAF→BSAF once, then for each (model_name, cfg) in `runs`:
      - build the DiscreteModular
      - solve
      - record strengths & convergence
    Finally put the list of entries into `queue`.
    """
    aba_path = Path(aba_path_str)
    entries = []

    try:
        # 1) load & build BSAF (heavy!)
        abaf = ABAF(path=str(aba_path))
        bsaf = abaf.to_bsaf()

        # 2) initial strengths
        initial_strengths = {
            a.name: a.initial_weight
            for a in bsaf.assumptions
        }

        # 3) for each model configuration
        for model_name, cfg in runs:
            entry = {
                "file":              aba_path.name,
                "file_path":         str(aba_path),
                "model":             model_name,
                **params,
                "initial_strengths": initial_strengths,
                "final_strengths":   None,
                "global_converged":  None,
                "prop_converged":    None,
                "per_arg":           None,
                "timeout":           False
            }

            # build & solve
            model = DiscreteModular(
                BSAF=bsaf,
                aggregation=cfg["aggregation"],
                influence=cfg["influence"],
                set_aggregation=cfg["set_aggregation"]
            )
            final_state = model.solve(20, generate_plot=True, verbose=False)

            # record final strengths
            final_strengths = {
                a.name: final_state[a]
                for a in model.assumptions
            }

            # convergence
            per_arg     = model.has_converged(epsilon=1e-3, last_n=5)
            global_conv = model.is_globally_converged(epsilon=1e-3, last_n=5)
            total       = len(per_arg)
            prop_conv   = (sum(per_arg.values())/total) if total else 0.0

            entry.update({
                "final_strengths":  final_strengths,
                "global_converged": global_conv,
                "prop_converged":   prop_conv,
                "per_arg":          per_arg
            })

            entries.append(entry)

    except Exception:
        # If *any* step fails (including a model.solve timeout via the main process kill),
        # mark *all* runs for this file as timed-out.
        for model_name, _ in runs:
            entries.append({
                "file":      aba_path.name,
                "file_path": str(aba_path),
                "model":     model_name,
                **params,
                "initial_strengths": None,
                "final_strengths":   None,
                "global_converged":  None,
                "prop_converged":    None,
                "per_arg":           None,
                "timeout":           True
            })

    # finally put list of two entries into queue
    queue.put(entries)


def run_file_with_timeout(aba_path, params, runs, timeout):
    """
    Spawn worker_file in its own process; if it doesn't finish in `timeout` secs,
    kill it and return two timeout‐only entries.
    Otherwise grab the two real entries from the queue.
    """
    q = Queue()
    p = Process(
        target=worker_file,
        args=(str(aba_path), params, runs, q)
    )
    p.start()
    p.join(timeout)

    if p.is_alive():
        # timed out: kill and return two timeout entries
        p.terminate()
        p.join()
        print(f"⚠️  Timeout on file {aba_path.name}")
        return q.get() if not q.empty() else [
            {
                "file":      aba_path.name,
                "file_path": str(aba_path),
                "model":     model_name,
                **params,
                "initial_strengths": None,
                "final_strengths":   None,
                "global_converged":  None,
                "prop_converged":    None,
                "per_arg":           None,
                "timeout":           True
            }
            for model_name, _ in runs
        ]

    # normal completion
    return q.get()


if __name__ == "__main__":
    results = []

    for aba_path in tqdm(aba_paths, desc="Files", unit="file"):
        # pull out s, c, n, a, r, b from the filename
        m = param_pat.search(aba_path.name)
        params = (
            dict(s=int(m.group("s")), c=float(m.group("c")),
                 n=float(m.group("n")), a=float(m.group("a")),
                 r=int(m.group("r")), b=int(m.group("b")))
            if m else dict(s=None, c=None, n=None, a=None, r=None, b=None)
        )

        # do both DF-QuAD and QE in one go, building arguments only once
        entries = run_file_with_timeout(
            aba_path, params, RUNS, TIMEOUT_SECONDS
        )
        results.extend(entries)

        # persist after each file
        with open(OUTPUT_PKL, "wb") as pf:
            pickle.dump(results, pf)

    print(f"✅  Complete — partial results saved in {OUTPUT_PKL}")
