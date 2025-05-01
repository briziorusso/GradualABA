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

INPUT_DIR       = Path("../dependency-graph-alternative/input_data_nf").resolve()
OUTPUT_PKL      = "convergence_results_test.pkl"
MAX_FILES       = 0
MAX_SENTENCES   = 25
MIN_SENTENCES   = 0
TIMEOUT_SECONDS = 10

# pick only .aba with s between MIN and MAX
pattern_s = re.compile(r"_s(\d+)_")
all_aba = sorted(INPUT_DIR.glob("*.aba"))
aba_paths = [p for p in all_aba
             if (m := pattern_s.search(p.name))
                and MIN_SENTENCES <= int(m.group(1)) <= MAX_SENTENCES]
if MAX_FILES > 0:
    aba_paths = aba_paths[:MAX_FILES]

# regex to extract s,c,n,a,r,b
param_pat = re.compile(
    r"_s(?P<s>\d+)_"
    r"c(?P<c>[\d.]+)_"
    r"n(?P<n>[\d.]+)_"
    r"a(?P<a>[\d.]+)_"
    r"r(?P<r>\d+)_"
    r"b(?P<b>\d+)"
)

def worker_run(aba_path_str, model_name, cfg, params, timeout, queue):
    """
    Runs one (file, model) end-to-end, writes `entry` dict into `queue`.
    """
    aba_path = Path(aba_path_str)
    entry = {
        "file": aba_path.name,
        "file_path": str(aba_path),
        "model": model_name,
        **params,
        "initial_strengths": None,
        "final_strengths":   None,
        "global_converged":  None,
        "prop_converged":    None,
        "per_arg":           None,
        "timeout":           False
    }
    try:
        # load and record initials
        abaf = ABAF(path=str(aba_path))
        bsaf = abaf.to_bsaf()
        init_strengths = {a.name: a.initial_weight for a in bsaf.assumptions}
        entry["initial_strengths"] = init_strengths

        # build model and solve
        model = DiscreteModular(
            BSAF=bsaf,
            aggregation=cfg["aggregation"],
            influence=cfg["influence"],
            set_aggregation=cfg["set_aggregation"]
        )
        final_state = model.solve(20, generate_plot=True, verbose=False)

        # record finals & convergence
        final_strengths = {a.name: final_state[a] for a in model.assumptions}
        per_arg = model.has_converged(epsilon=1e-3, last_n=5)
        global_conv = model.is_globally_converged(epsilon=1e-3, last_n=5)
        total = len(per_arg)
        prop_conv = sum(per_arg.values()) / total if total else 0.0

        entry.update({
            "final_strengths":  final_strengths,
            "global_converged": global_conv,
            "prop_converged":   prop_conv,
            "per_arg":          per_arg
        })
    except Exception as e:
        # if anything blows up (including timeout), mark it
        entry["timeout"] = True

    queue.put(entry)


def run_with_timeout(aba_path, model_name, cfg, params, timeout):
    q = Queue()
    p = Process(
        target=worker_run,
        args=(str(aba_path), model_name, cfg, params, timeout, q)
    )
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        # give back a timeout-only entry
        print(f"Timeout on {aba_path.name} ({model_name})")
        return {
            "file": aba_path.name,
            "file_path": str(aba_path),
            "model": model_name,
            **params,
            "initial_strengths": None,
            "final_strengths":   None,
            "global_converged":  None,
            "prop_converged":    None,
            "per_arg":           None,
            "timeout":           True
        }
    # otherwise grab result
    return q.get()


if __name__ == "__main__":
    results = []

    for aba_path in tqdm(aba_paths, desc="Files", unit="file"):
        # parse params
        m = param_pat.search(aba_path.name)
        params = (
            dict(s=int(m.group("s")), c=float(m.group("c")),
                 n=float(m.group("n")), a=float(m.group("a")),
                 r=int(m.group("r")), b=int(m.group("b")))
            if m else dict(s=None, c=None, n=None, a=None, r=None, b=None)
        )

        runs = [
            ("DF-QuAD", {
                "aggregation":    ProductAggregation(),
                "influence":      LinearInfluence(conservativeness=1),
                "set_aggregation": SetProductAggregation()
            }),
            ("QE", {
                "aggregation":    SumAggregation(),
                "influence":      QuadraticMaximumInfluence(conservativeness=1),
                "set_aggregation": SetProductAggregation()
            })
        ]

        for model_name, cfg in runs:
            entry = run_with_timeout(aba_path, model_name, cfg, params, TIMEOUT_SECONDS)
            results.append(entry)

        # save after each file
        with open(OUTPUT_PKL, "wb") as pf:
            pickle.dump(results, pf)

    print(f"Complete — partial results in {OUTPUT_PKL}")
