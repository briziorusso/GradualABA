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

# ─── Config ──────────────────────────────────────────────────────────────
INPUT_DIR       = Path("data_generation/abaf/").resolve()
OUTPUT_PKL      = "convergence_results_to10m_nf_atm_flatrerun.pkl"
MAX_FILES       = 0       # 0 = no limit
MIN_SENTENCES   = 0
MAX_SENTENCES   = 100
TIMEOUT_SECONDS = 600      # per‐file timeout

def is_disk_flat(p: Path) -> bool:
    lines = p.read_text().splitlines()
    assumps = {parts[1] for ln in lines if ln.startswith("a ") for parts in [ln.split()] if len(parts)>1}
    for ln in lines:
        if not ln.startswith("r "):
            continue
        parts = ln.split()
        if len(parts) >= 2 and parts[1] in assumps:
            return False
    return True

pattern_s = re.compile(r"_s(\d+)_")
all_aba   = sorted(INPUT_DIR.glob("*.aba"))
aba_paths = [
    p for p in all_aba
    if (m := pattern_s.search(p.name))
       and MIN_SENTENCES <= int(m.group(1)) <= MAX_SENTENCES
       and is_disk_flat(p)
]
if MAX_FILES > 0:
    aba_paths = aba_paths[:MAX_FILES]

param_pat = re.compile(
    r"_s(?P<s>\d+)_"
    r"n(?P<n>[\d.]+)_"
    r"a(?P<a>[\d.]+)_"
    r"r(?P<r>\d+)_"
    r"b(?P<b>\d+)"
)

RUNS = [
    ("DF-QuAD", dict(
        aggregation     = ProductAggregation(),
        influence       = LinearInfluence(conservativeness=1),
        set_aggregation = SetProductAggregation()
    )),
    ("QE",      dict(
        aggregation     = SumAggregation(),
        influence       = QuadraticMaximumInfluence(conservativeness=1),
        set_aggregation = SetProductAggregation()
    ))
]

def disk_non_flat(path: Path) -> bool:
    """
    Quickly scan the .aba file to see if any rule-head is an assumption.
    Mirrors _load_from_file's 'a ' / 'r ' logic.
    """
    lines = path.read_text().splitlines()
    # collect assumption names
    assumptions = {
        parts[1]
        for line in lines if line.startswith("a ")
        for parts in [line.split()]
        if len(parts) >= 2
    }
    # if any rule head appears in assumptions => non-flat
    for line in lines:
        if not line.startswith("r "):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        head = parts[1]
        if head in assumptions:
            return True
    return False

def worker_file(aba_path_str, params, runs, queue):
    """
    Runs one file end-to-end, reporting both DF-QuAD and QE entries.
    Any exception here will now crash the subprocess.
    """
    aba_path    = Path(aba_path_str)
    is_non_flat = disk_non_flat(aba_path)

    # 1) load & build BSAF (heavy!)
    abaf = ABAF(path=str(aba_path))
    bsaf = abaf.to_bsaf()

    # 2) compute size‐stats
    num_assumptions = len(abaf.assumptions)
    num_rules       = len(abaf.rules)
    num_sentences   = len(abaf.sentences)

    # 3) initial strengths
    initial_strengths = { a.name: a.initial_weight for a in bsaf.assumptions }

    entries = []
    # 4) for each model configuration
    for model_name, cfg in runs:
        entry = {
            "file":              aba_path.name,
            "file_path":         str(aba_path),
            "model":             model_name,
            **params,
            "num_assumptions":   num_assumptions,
            "num_rules":         num_rules,
            "num_sentences":     num_sentences,
            "non_flat":          is_non_flat,
            "initial_strengths": initial_strengths,
            "final_strengths":   None,
            "global_converged":  None,
            "prop_converged":    None,
            "per_arg":           None,
            "timeout":           False
        }

        # build & solve
        model       = DiscreteModular(
                          BSAF=bsaf,
                          aggregation=cfg["aggregation"],
                          influence=cfg["influence"],
                          set_aggregation=cfg["set_aggregation"]
                      )
        final_state = model.solve(20, generate_plot=True, verbose=False)

        # record final strengths & convergence
        final_strengths = {a.name: final_state[a] for a in model.assumptions}
        per_arg         = model.has_converged(epsilon=1e-3, last_n=5)
        global_conv     = model.is_globally_converged(epsilon=1e-3, last_n=5)
        total           = len(per_arg)
        prop_conv       = (sum(per_arg.values()) / total) if total else 0.0

        entry.update({
            "final_strengths":  final_strengths,
            "global_converged": global_conv,
            "prop_converged":   prop_conv,
            "per_arg":          per_arg
        })

        entries.append(entry)

    # only on successful completion do we push entries
    queue.put(entries)


def run_file_with_timeout(aba_path, params, runs, timeout):
    """
    Spawn worker_file in its own process; if it doesn't finish in `timeout` secs,
    kill it and return two timeout‐only entries.
    If it exits with an error, raise in the main process.
    """
    q = Queue()
    p = Process(target=worker_file, args=(str(aba_path), params, runs, q))
    p.start()
    p.join(timeout)

    if p.is_alive():
        # timed out: kill and fallback
        p.terminate()
        p.join()
        print(f"⚠️  Timeout on file {aba_path.name}")
        return [
            {
                "file":      aba_path.name,
                "file_path": str(aba_path),
                "model":     model_name,
                **params,
                "timeout":   True,
                "non_flat":  disk_non_flat(aba_path)
            }
            for model_name, _ in runs
        ]

    # process is no longer alive
    if p.exitcode != 0:
        # crash in worker_file → propagate
        raise RuntimeError(f"Worker for {aba_path.name} crashed with exitcode {p.exitcode}")

    # successful run, grab the two entries
    return q.get()


if __name__ == "__main__":
    results = []
    for aba_path in tqdm(aba_paths, desc="Files", unit="file"):
        # extract s,n,a,r,b from filename
        m = param_pat.search(aba_path.name)
        params = (
            dict(
              s=int(m.group("s")),
              n=float(m.group("n")),
              a=float(m.group("a")),
              r=int(m.group("r")),
              b=int(m.group("b"))
            ) if m else dict(s=None, n=None, a=None, r=None, b=None)
        )

        # run both DF-QuAD and QE, building arguments only once
        entries = run_file_with_timeout(
            aba_path, params, RUNS, TIMEOUT_SECONDS
        )
        results.extend(entries)

        # persist after each file
        with open(OUTPUT_PKL, "wb") as pf:
            pickle.dump(results, pf)

    print(f"✅  Complete — results saved in {OUTPUT_PKL}")
