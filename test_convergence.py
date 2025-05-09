import re
import pickle
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Process, Queue
from collections import defaultdict
import random
import traceback, sys
from decimal import Decimal
import signal
import sys
sys.path.append("../")

from ABAF import ABAF
from ABAF.Assumption import Assumption
from BSAF.Argument import Argument

from semantics.bsafDiscreteModular import DiscreteModular
from semantics.DiscreteModularModel import DiscreteModularBAG
from semantics.modular.ProductAggregation    import ProductAggregation
from semantics.modular.SetProductAggregation import SetProductAggregation
from semantics.modular.SetMinAggregation     import SetMinAggregation
from semantics.modular.SetMeanAggregation    import SetMeanAggregation
from semantics.modular.SumAggregation        import SumAggregation
from semantics.modular.LinearInfluence       import LinearInfluence
from semantics.modular.QuadraticMaximumInfluence import QuadraticMaximumInfluence

# ─── Config ──────────────────────────────────────────────────────────────
INPUT_DIR       = Path("data_generation/abaf/").resolve()
CACHE_DIR       = Path(INPUT_DIR,"bsaf_frameworks")
CACHE_DIR_BAG   = Path(INPUT_DIR,"bag_frameworks")
CACHE_OVERRIDE  = False # set to True to override existing cache files
RESULT_OVERRIDE = False # set to True to override existing results

SEED            = 42      # random seed for reproducibility
MAX_FILES       = 0       # 0 = no limit
MIN_SENTENCES   = 0
MAX_SENTENCES   = 100
TIMEOUT_SECONDS = 600     # per‐file timeout
EPSILON         = 1e-3    # convergence epsilon
DELTA           = 5       # convergence delta
MAX_STEPS       = 5000    # max steps for convergence
BASE_SCORES     = 'random' # 'random' or '' (empty==DEFAULT_WEIGHTS)
SET_AGGREGATION = SetProductAggregation() # SetProductAggregation() or SetMinAggregation()
ASM_AGGREGATION  = 'SelectAsmArguments' # SetMeanAggregation() or SelectAsmArguments or SetMinAggregation()
# ────────────────────────────────────────────────────────────────────────

## combine parameters into out name for the output file
set_agg_name = 'prod' if isinstance(SET_AGGREGATION, SetProductAggregation) else 'min'
if isinstance(ASM_AGGREGATION, SetMeanAggregation):
    asm_agg_name = 'mean'
elif isinstance(ASM_AGGREGATION, SetMinAggregation):
    asm_agg_name = 'min'
elif ASM_AGGREGATION == 'SelectAsmArguments':
    asm_agg_name = 'sel'
else:
    raise ValueError(f"Unknown ASM aggregation: {ASM_AGGREGATION}")

base_init = '_randinitall' if BASE_SCORES == 'random' else ''

out_name = f"convergence_results_to{int(TIMEOUT_SECONDS / 60)}m_nf_atm_e{str('%.e' % Decimal(EPSILON))[-1]}_d{DELTA}_s{MAX_STEPS}{base_init}_{set_agg_name}_{asm_agg_name}.pkl"

OUTPUT_PKL = Path(out_name)

TIMEOUT_RECORD  = CACHE_DIR / f"timed_out_{TIMEOUT_SECONDS}s.txt"
TIMEOUT_RECORD.touch(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
CACHE_DIR_BAG.mkdir(exist_ok=True)
# random.seed(SEED)

# ─── 0) LOAD the set of already‐timed‐out stems ─────────────────────────
with open(TIMEOUT_RECORD, "r") as f:
    timed_out_stems = { line.strip() for line in f if line.strip() }

# 1) if OUTPUT_PKL exists, load it; otherwise start empty
if OUTPUT_PKL.exists():
    with open(OUTPUT_PKL, "rb") as f:
        results = pickle.load(f)
else:
    results = []

by_file = defaultdict(list)
for r in results:
    by_file[ Path(r["file"]).stem ].append(r)

pattern_s = re.compile(r"_s(\d+)_")
all_aba = sorted(INPUT_DIR.glob("*.aba"))
aba_paths = [
    p for p in all_aba
    if (m := pattern_s.search(p.name))
       and MIN_SENTENCES <= int(m.group(1)) <= MAX_SENTENCES
       and p.stem not in timed_out_stems     # skip timed‐out files
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

# RUNS = [
#     ("DF-QuAD", dict(
#         aggregation     = ProductAggregation(),
#         influence       = LinearInfluence(conservativeness=1),
#         set_aggregation = SET_AGGREGATION
#     )),
#     ("QE", dict(
#         aggregation     = SumAggregation(),
#         influence       = QuadraticMaximumInfluence(conservativeness=1),
#         set_aggregation = SET_AGGREGATION
#     )),
# ]

BSAF_RUNS = [
    # ("DF-QuAD (BSAF)", dict(
    #     klass        = DiscreteModular,
    #     aggregation  = ProductAggregation(),
    #     influence    = LinearInfluence(conservativeness=1),
    #     set_aggregation = SET_AGGREGATION
    # )),
    # ("QE      (BSAF)", dict(
    #     klass        = DiscreteModular,
    #     aggregation  = SumAggregation(),
    #     influence    = QuadraticMaximumInfluence(conservativeness=1),
    #     set_aggregation = SET_AGGREGATION
    # )),
]

BAG_RUNS = [
    ("DF-QuAD (BAF)", dict(
        klass        = DiscreteModularBAG,
        aggregation  = ProductAggregation(),
        influence    = LinearInfluence(conservativeness=1),
        set_aggregation = SET_AGGREGATION,
        ams_aggregation = ASM_AGGREGATION
    )),
    ("QE (BAF)", dict(
        klass        = DiscreteModularBAG,
        aggregation  = SumAggregation(),
        influence    = QuadraticMaximumInfluence(conservativeness=1),
        set_aggregation = SET_AGGREGATION,
        ams_aggregation = ASM_AGGREGATION
    )),
]

ALL_RUNS = [
    *BSAF_RUNS,
    *BAG_RUNS,
]


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

def disk_non_flat(path: Path) -> bool:
    return not is_disk_flat(path)

def load_or_build_bsaf(aba_path: Path):
    """
    Caches ABAF.to_bsaf() on disk under bsaf_cache/<stem>.bsaf.pkl.
    If CACHE_OVERRIDE is True, always rebuild (even if the cache file exists).
    """

    Argument.reset_identifiers()  # reset the identifiers for the next run
    Assumption.reset_identifiers()  # reset the identifiers for the next run

    cache_file = CACHE_DIR / (aba_path.stem + f"{BASE_SCORES}.bsaf.pkl")

    # if cache exists and we're not overriding, just load
    if cache_file.exists() and not CACHE_OVERRIDE:
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except EOFError:
            # truncated or empty cache: remove and fall through to rebuild
            print(f"[CACHE CORRUPT] {cache_file.name}, rebuilding…", flush=True)
            cache_file.unlink()

    # otherwise we're (re)building
    if cache_file.exists() and CACHE_OVERRIDE:
        print(f"[OVERRIDE]    {aba_path.name}  (rebuilding cache)", flush=True)
    else:
        print(f"[BUILDING]    {aba_path.name}", flush=True)

    # build & cache
    abaf = ABAF(path=str(aba_path))
    bsaf = abaf.to_bsaf()
    with open(cache_file, "wb") as f:
        pickle.dump(bsaf, f)
    return bsaf

def load_or_build_bag(aba_path: Path, weight_agg, args=None, abaf=None):
    """
    Caches ABAF.to_bag() on disk under bag_cache/<stem>.bag.pkl.
    If CACHE_OVERRIDE is True, always rebuild (even if the cache file exists).
    """
    Argument.reset_identifiers()
    Assumption.reset_identifiers()

    cache_file = CACHE_DIR_BAG / (aba_path.stem + f"{BASE_SCORES}{weight_agg.name}.bag.pkl")

    # 1) If cache exists and we're not forcing an override, try to load it
    if cache_file.exists() and not CACHE_OVERRIDE:
        print(f"[CACHE HIT]   {aba_path.name}", flush=True)
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except:
            # truncated or empty cache: remove and fall through to rebuild
            print(f"[CACHE CORRUPT] {cache_file.name}, rebuilding…", flush=True)
            cache_file.unlink()

    # 2) If we're explicitly overriding, or if the file didn't exist (or we just deleted it),
    #    rebuild the BAG and re-cache it.
    if cache_file.exists() and CACHE_OVERRIDE:
        print(f"[OVERRIDE]    {aba_path.name}  (rebuilding cache)", flush=True)
    elif not cache_file.exists():
        print(f"[BUILDING]    {aba_path.name}", flush=True)

    # build & cache
    if abaf is None:
        abaf = ABAF(path=str(aba_path))

    # respect whether args were provided
    if args is None:
        bag = abaf.to_bag(weight_agg=weight_agg)
    else:
        bag = abaf.to_bag(weight_agg=weight_agg, args=args)

    # write out a fresh cache
    with open(cache_file, "wb") as f:
        pickle.dump(bag, f)

    return bag

# 3) helper to decide whether to skip this file
def should_skip(path: Path):
    stem = path.stem
    entries = by_file.get(stem, [])
    # if we've never run it, don't skip
    if not entries:
        return False
    # if override_all, don't skip
    if RESULT_OVERRIDE:
        return False
    # by default (no override flags) skip if we have 2 entries already
    return len(entries) >= 2

def worker_file(aba_path_str, params, runs, queue):
    """
    Runs one file end-to-end, reporting both DF-QuAD and QE entries.
    Any exception here will now crash the subprocess.
    """
    try:
        aba_path    = Path(aba_path_str)
        is_non_flat = disk_non_flat(aba_path)

        # 1) load or build & cache the BSAF
        bsaf = load_or_build_bsaf(aba_path)

        # 2) re-load ABAF only to count size-stats and update the initial strengths
        abaf = ABAF(path=str(aba_path), 
                    weight_fn=(lambda: random.uniform(0.0, 1.0)) if BASE_SCORES == "random" else None)
        num_assumptions = len(abaf.assumptions)
        num_rules       = len(abaf.rules)
        num_sentences   = len(abaf.sentences)

        # 3) collect initial strengths
        initial_strengths = { a.name: a.initial_weight for a in abaf.assumptions}

        ## update the initial strengths in the BSAF
        for a in bsaf.assumptions:
            if a.name in initial_strengths:
                a.initial_weight = initial_strengths[a.name]
            else:
                raise ValueError(f"Assumption {a.name} not found in initial strengths")

        entries = []
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
                "convergence_time":  None,
                "timeout":           False
            }

            # build & solve
            # model       = DiscreteModular(
            #                 BSAF=bsaf,
            #                 aggregation=cfg["aggregation"],
            #                 influence=cfg["influence"],
            #                 set_aggregation=cfg["set_aggregation"]
            #             )
            # final_state = model.solve(MAX_STEPS, generate_plot=True, verbose=False)

            # # pick the right constructor and input
            if cfg["klass"] is DiscreteModular:
                # BSAF‐based
                approach = "BSAF"
                model = cfg["klass"](
                    BSAF=bsaf,
                    aggregation=cfg["aggregation"],
                    influence=cfg["influence"],
                    set_aggregation=cfg["set_aggregation"]
                )
                final_state = model.solve(
                    MAX_STEPS,
                    generate_plot=True,
                    verbose=False
                )
            else:  # BAG‐based
                approach = "BAF"
                # bag = abaf.to_bag(weight_agg=cfg["set_aggregation"], args=bsaf.arguments)
                bag = load_or_build_bag(Path(aba_path), weight_agg=cfg["set_aggregation"], args=bsaf.arguments, abaf=abaf)
                print(bag)
                model = cfg["klass"](
                    bag=bag,
                    aggregation=cfg["aggregation"],
                    influence=cfg["influence"],
                    set_aggregation=cfg["set_aggregation"]
                )
                final_state = model.solve(
                    MAX_STEPS,
                    generate_plot=True,
                    verbose=False,
                    view="Assumptions",
                    assumptions=bsaf.assumptions,
                    aggregate_strength_f=cfg["ams_aggregation"]                
                )

            # record metrics
            final_strengths = {a.name: final_state[a] for a in model.assumptions}
            per_arg         = model.has_converged(epsilon=EPSILON, last_n=DELTA)
            global_conv     = model.is_globally_converged(epsilon=EPSILON, last_n=DELTA)
            conv_time       = model.convergence_time(epsilon=EPSILON, consecutive=DELTA)
            total           = len(per_arg)
            prop_conv       = (sum(per_arg.values()) / total) if total else 0.0

            entry.update({
                "final_strengths":  final_strengths,
                "global_converged": global_conv,
                "prop_converged":   prop_conv,
                "per_arg":          per_arg,
                "convergence_time": conv_time,
            })

            entries.append(entry)

        queue.put(entries)
    
    except Exception as e:
        tb = traceback.format_exc()
        queue.put({"__error__": tb})
        # now exit cleanly
        return

def run_file_with_timeout(aba_path, params, runs, timeout):
    q = Queue()
    p = Process(target=worker_file, args=(str(aba_path), params, runs, q))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        print(f"⚠️  Timeout on file {aba_path.name}")
        return [
            {
                "file":      aba_path.name,
                "file_path": str(aba_path),
                "model":     model_name,
                **params,
                "convergence_time": None,
                "timeout":         True,
                "non_flat":        disk_non_flat(aba_path)
            }
            for model_name, _ in runs
        ]
    
    # 2) child died, see if it sent us an error
    if not q.empty():
        msg = q.get()
        if isinstance(msg, dict) and "__error__" in msg:
            print(f"\n⚠️  Worker crashed on {aba_path.name}:\n")
            print(msg["__error__"], file=sys.stderr)
            raise RuntimeError(f"Worker for {aba_path.name} raised an exception")

    if p.exitcode == -signal.SIGTERM: 
    ## This does not catch if it runs out of memory because of this file or
    # another file running at the same time
        print(f"⚠️  OOM on file {aba_path.name}")
        return [
            {
                "file":      aba_path.name,
                "file_path": str(aba_path),
                "model":     model_name,
                **params,
                "convergence_time": None,
                "timeout":         False,
                "oom":             True,
                "non_flat":        disk_non_flat(aba_path)
            }
            for model_name, _ in runs
        ]
    
    # 3) if child exitcode != 0 but no traceback, still bail
    if p.exitcode != 0:
        raise RuntimeError(f"Worker for {aba_path.name} exited with code {p.exitcode} and no traceback. \n \
                           Run: {aba_path.name}")

    # 4) otherwise we got a normal payload: our two entries
    return msg  # this is the 'entries' list from the child


if __name__ == "__main__":
    for aba_path in tqdm(aba_paths, desc="Files", unit="file"):
        # skip if we've already run this file
        if should_skip(aba_path):
            print(f"⚠️  Skipping {aba_path.name} (already run)", flush=True)
            continue
        # parse params from filename
        m = param_pat.search(aba_path.name)
        params = (dict(
              s=int(m.group("s")),
              n=float(m.group("n")),
              a=float(m.group("a")),
              r=int(m.group("r")),
              b=int(m.group("b"))
        ) if m else dict(s=None,n=None,a=None,r=None,b=None))

        print(f"\n=== Running on {aba_path.name}, Scenario: {BASE_SCORES, set_agg_name, asm_agg_name} ===", flush=True)

        entries = run_file_with_timeout(aba_path, params, ALL_RUNS, TIMEOUT_SECONDS)
        results.extend(entries)

        # persist after each file
        with open(OUTPUT_PKL, "wb") as pf:
            pickle.dump(results, pf)

    print(f"✅  Complete — results in {OUTPUT_PKL}")
