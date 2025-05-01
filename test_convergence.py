import re
import pickle
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append("../")

from ABAF import ABAF
from semantics.bsafDiscreteModular import DiscreteModular

from semantics.modular.ProductAggregation        import ProductAggregation
from semantics.modular.SetProductAggregation     import SetProductAggregation
from semantics.modular.SumAggregation            import SumAggregation
from semantics.modular.LinearInfluence           import LinearInfluence
from semantics.modular.QuadraticMaximumInfluence import QuadraticMaximumInfluence

INPUT_DIR   = Path("../dependency-graph-alternative/input_data_nf").resolve()
OUTPUT_PKL  = "convergence_results.pkl"
MAX_FILES   = 0 # 0 = all
MAX_SENTENCES = 25
MIN_SENTENCES = 0

# 1) pick only .aba with sample-size "s≤MAX_SENTENCES"
pattern_s = re.compile(r"_s(\d+)_")
all_aba = sorted(INPUT_DIR.glob("*.aba"))
aba_paths = [p for p in all_aba
             if (m := pattern_s.search(p.name)) and int(m.group(1)) <= MAX_SENTENCES and int(m.group(1)) >= MIN_SENTENCES]

# limit number of files to process for testing
if MAX_FILES > 0:
    aba_paths = aba_paths[:MAX_FILES]

# 2) regex to extract all parameters from filename
param_pat = re.compile(
    r"_s(?P<s>\d+)_"
    r"c(?P<c>[\d.]+)_"
    r"n(?P<n>[\d.]+)_"
    r"a(?P<a>[\d.]+)_"
    r"r(?P<r>\d+)_"
    r"b(?P<b>\d+)"
)

results = []

for aba_path in tqdm(aba_paths, desc="Files", unit="file"):
    # extract parameters
    m = param_pat.search(aba_path.name)
    if not m:
        # skip or fill with None
        params = dict(s=None, c=None, n=None, a=None, r=None, b=None)
    else:
        params = {
            "s": int(m.group("s")),
            "c": float(m.group("c")),
            "n": float(m.group("n")),
            "a": float(m.group("a")),
            "r": int(m.group("r")),
            "b": int(m.group("b")),
        }

    # load ABAF → BSAF
    abaf = ABAF(path=str(aba_path))
    bsaf = abaf.to_bsaf()

    # record initial strengths once per file
    initial_strengths = { asm.name: asm.initial_weight 
                          for asm in bsaf.assumptions }

    # two models to run
    runs = [
        ("DF-QuAD", {"aggregation": ProductAggregation(),
                     "influence": LinearInfluence(conservativeness=1),
                     "set_aggregation": SetProductAggregation()}),
        ("QE",      {"aggregation": SumAggregation(),
                     "influence": QuadraticMaximumInfluence(conservativeness=1),
                     "set_aggregation": SetProductAggregation()})
    ]

    for model_name, cfg in runs:
        # build & solve
        model = DiscreteModular(
            BSAF=bsaf,
            aggregation=cfg["aggregation"],
            influence=cfg["influence"],
            set_aggregation=cfg["set_aggregation"]
        )
        final_state = model.solve(20, generate_plot=True)

        # map final strengths by name
        final_strengths = { asm.name: final_state[asm] 
                            for asm in model.assumptions }

        # convergence checks
        per_arg     = model.has_converged(epsilon=1e-3, last_n=5)
        global_conv = model.is_globally_converged(epsilon=1e-3, last_n=5)
        total       = len(per_arg)
        num_conv    = sum(per_arg.values())
        prop_conv   = (num_conv / total) if total else 0.0

        # record everything
        entry = {
            "file":              aba_path.name,
            "file_path":         str(aba_path),
            "model":             model_name,
            **params,  # s, c, n, a, r, b
            "initial_strengths": initial_strengths,
            "final_strengths":   final_strengths,
            "global_converged":  global_conv,
            "prop_converged":    prop_conv,
            "per_arg":           per_arg
        }
        results.append(entry)

    # persist after each file
    with open(OUTPUT_PKL, "wb") as pf:
        pickle.dump(results, pf)

print(f"Done! Results up to this point saved in {OUTPUT_PKL}")
