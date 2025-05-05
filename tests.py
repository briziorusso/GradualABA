import unittest
import os, sys
import re
import random
import pickle
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Process, Queue
from collections import defaultdict
import sys
sys.path.append("../")

from ABAF import ABAF
from ABAF.Assumption import Assumption
from ABAF.Rule import Rule
from ABAF.Sentence import Sentence
from ABAF.ABAF import ABAF

from BSAF.Argument import Argument
from BSAF.BSAF import BSAF
from BAG.BAG import BAG

## Common components
from semantics.modular.ProductAggregation        import ProductAggregation
from semantics.modular.SumAggregation            import SumAggregation
from semantics.modular.LinearInfluence           import LinearInfluence
from semantics.modular.QuadraticMaximumInfluence import QuadraticMaximumInfluence

## BSAF components
from semantics.bsafDiscreteModular import DiscreteModular
from semantics.modular.SetProductAggregation import SetProductAggregation
from semantics.modular.SetMinAggregation import SetMinAggregation

## BAG
from semantics.DiscreteModularModel import DiscreteModularBAG

class TestBSAF(unittest.TestCase):

    def test_aba_bsaf_1(self):

        #### BAG
        ### Arguments
        ## ({a}, a)
        ## ({b}, b)
        ## ({c}, c)
        ## ({d}, d)
        ## ({d}, e)
        ## ({b,c}, f)
        ## ({d,c}, b)
        ## ({d,c}, f)

        ### Supports
        ## ({d,c}, b) -+ ({b}, b)

        ### Attacks
        ## ({b,c}, f) -> ({a}, a)
        ## ({d,c}, f) -> ({a}, a)

        #### BSAF
        ### SetSupports
        ## {d,c} -+ b

        ### SetAttacks
        ## {b,c} -> a
        ## {d,c} -> a


        ## Assumptions
        a = Assumption("a", initial_weight=0.8)
        b = Assumption("b", initial_weight=0.5)
        c = Assumption("c", initial_weight=0.5)
        d = Assumption("d", initial_weight=0.5)
        e = Sentence("e")
        f = Sentence("f")
        g = Sentence("g")

        ## Contraries
        a.contrary = 'f'

        ## Rules
        r1 = Rule(head=b, body=[e,c], name="r1")
        r2 = Rule(head=e, body=[d], name="r2")
        r3 = Rule(head=f, body=[b,c], name="r3")

        abaf = ABAF(assumptions=[a,b,c,d], rules=[r1,r2,r3])

        print(abaf)

        bsaf = abaf.to_bsaf()

        ## extract attacks and supports and arguments
        nonemptyattacks = set()
        nonemptysupports = set()
        arglist = set()
        for asm in bsaf.assumptions:
            for att in bsaf.attacks[asm]:
                attacked = asm.name
                attackers = frozenset(a.name for a in att)
                nonemptyattacks.add((attackers, attacked))
        for asm in bsaf.assumptions:
            for supp in bsaf.supports[asm]:
                supported = asm.name
                supporters = frozenset(a.name for a in supp)
                if len(supporters) > 1:
                    nonemptysupports.add((supporters, supported))
        for arg in bsaf.arguments:
            arglist.add((frozenset(a.name for a in arg.premise), arg.claim.name))
            
        expected_attacks = set([
            (frozenset(['b', 'c']), 'a'),
            (frozenset(['d', 'c']), 'a')
        ])
        expected_supports = set([
            (frozenset(['d', 'c']), 'b')
        ])
        expected_arguments = set([
            (frozenset(['a']), 'a'),
            (frozenset(['b']), 'b'),
            (frozenset(['c']), 'c'),
            (frozenset(['d']), 'd'),
            (frozenset(['d']), 'e'),
            (frozenset(['b', 'c']), 'f'),
            (frozenset(['d', 'c']), 'b'),
            (frozenset(['d', 'c']), 'f')
        ])

        # Check if the extracted attacks and supports match the expected values
        self.assertEqual(nonemptyattacks, expected_attacks, f"Expected attacks: {expected_attacks}, but got: {nonemptyattacks}")
        self.assertEqual(nonemptysupports, expected_supports, f"Expected supports: {expected_supports}, but got: {nonemptysupports}")
        self.assertEqual(arglist, expected_arguments, f"Expected arguments: {expected_arguments}, but got: {arglist}")

        print("Arguments, attacks and supports extracted correctly.")

    def test_aba_bsaf_2(self):

        Rule.reset_identifiers()
        
        ## Assumptions
        a = Assumption("a", initial_weight=0.8)
        b = Assumption("b", initial_weight=0.5)
        c = Assumption("c", initial_weight=0.5)
        d = Assumption("d", initial_weight=0.5)
        e = Sentence("e")
        f = Sentence("f")
        g = Sentence("g")

        ## Contraries
        a.contrary = 'f'

        ## Rules
        r1 = Rule(head=f, body=[e,g], name="r1")
        r2 = Rule(head=e, body=[c], name="r2")
        r3 = Rule(head=e, body=[d,a], name="r3")
        r4 = Rule(head=g, body=[b,a], name="r4")
        r5 = Rule(head=e, body=[a], name="r5")

        abaf = ABAF(assumptions=[a,b,c,d], rules=[r4,r2,r1,r3,r5])

        bsaf = abaf.to_bsaf()

        ## extract attacks and supports and arguments
        nonemptyattacks = set()
        nonemptysupports = set()
        arglist = set()
        for asm in bsaf.assumptions:
            for att in bsaf.attacks[asm]:
                attacked = asm.name
                attackers = frozenset(a.name for a in att)
                nonemptyattacks.add((attackers, attacked))
        for asm in bsaf.assumptions:
            for supp in bsaf.supports[asm]:
                supported = asm.name
                supporters = frozenset(a.name for a in supp)
                if len(supporters) > 1:
                    nonemptysupports.add((supporters, supported))
        for arg in bsaf.arguments:
            arglist.add((frozenset(a.name for a in arg.premise), arg.claim.name))

        
        expected_attacks = set([
            (frozenset({'a', 'b'}), 'a'),
            (frozenset({'a', 'c', 'b'}), 'a'),
            (frozenset({'a', 'd', 'b'}), 'a')
        ])

        expected_supports = set([

        ])

        expected_arguments = set([
            (frozenset(['a']), 'a'),
            (frozenset(['b']), 'b'),
            (frozenset(['c']), 'c'),
            (frozenset(['d']), 'd'),
            (frozenset({'b', 'a'}), 'f'),
            (frozenset({'a', 'd'}), 'e'),
            (frozenset({'c'}), 'e'),
            (frozenset({'b', 'a', 'd'}), 'f'),
            (frozenset({'a'}), 'e'),
            (frozenset({'b', 'c', 'a'}), 'f'),
            (frozenset({'b', 'a'}), 'g')
        ])

        # Check if the extracted attacks and supports match the expected values
        self.assertEqual(nonemptyattacks, expected_attacks, f"Expected attacks: {expected_attacks}, but got: {nonemptyattacks}")
        self.assertEqual(nonemptysupports, expected_supports, f"Expected supports: {expected_supports}, but got: {nonemptysupports}")
        self.assertEqual(arglist, expected_arguments, f"Expected arguments: {expected_arguments}, but got: {arglist}")

        print("Arguments, attacks and supports extracted correctly.")

    def test_convergence_random_weight(self):
        """
        Test the convergence of a single file with random weights.
        """
        # ─── Config ──────────────────────────────────────────────────────────────
        INPUT_DIR       = Path("data_generation/abaf/").resolve()
        MAX_FILES       = 0       # 0 = no limit
        MIN_SENTENCES   = 0
        MAX_SENTENCES   = 100
        BASE_SCORES     = "random"
        MAX_STEPS       = 20
        EPSILON         = 1e-3
        DELTA           = 5
        SEED           = 42

        random.seed(SEED)

        pattern_s = re.compile(r"_s(\d+)_")
        all_aba   = sorted(INPUT_DIR.glob("*.aba"))
        aba_paths = [
            p for p in all_aba
            if (m := pattern_s.search(p.name))
            and MIN_SENTENCES <= int(m.group(1)) <= MAX_SENTENCES
        ]
        if MAX_FILES > 0:
            aba_paths = aba_paths[:MAX_FILES]

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

        aba_path = "data_generation/abaf/nf_atm_s20_n0.01_a0.5_r2_b16_0.aba"

        abaf = ABAF(path=str(aba_path), 
                    weight_fn=lambda: round(random.uniform(0.0, 1.0),3) if BASE_SCORES == "random" else None)
        print(f"Loaded {aba_path.split('/')[-1]}. Flat: {not abaf.non_flat}")
        bsaf = abaf.to_bsaf()  # this sets abaf.non_flat, but we trust disk flag

        # 4) for each model configuration
        for model_name, cfg in RUNS:

            # build & solve
            model       = DiscreteModular(
                            BSAF=bsaf,
                            aggregation=cfg["aggregation"],
                            influence=cfg["influence"],
                            set_aggregation=cfg["set_aggregation"]
                        )
            final_state = model.solve(MAX_STEPS, generate_plot=True, verbose=False)

            # record final strengths & convergence
            per_arg         = model.has_converged(epsilon=EPSILON, last_n=DELTA)
            global_conv     = model.is_globally_converged(epsilon=EPSILON, last_n=DELTA)
            conv_time       = model.convergence_time(epsilon=EPSILON, consecutive=DELTA, out_mean=True)
            total           = len(per_arg)
            prop_conv       = (sum(per_arg.values()) / total) if total else 0.0
        
            print(f"Model: {model_name}, Global Convergence: {global_conv}, Proportion Converged: {prop_conv}, Convergence Time: {conv_time}")

        ### extract initial and final strengths
        initial_strengths = {a.name: a.initial_weight for a in model.assumptions}
        final_strengths   = {a.name: final_state[a] for a in model.assumptions}
        print(f"Initial strengths: {initial_strengths}")
        print(f"Final strengths: {final_strengths}")

        expected_initial_strengths = {'a2': 0.784, 'a4': 0.621, 'a1': 0.964, 'a5': 0.79, 'a0': 0.02, 'a9': 0.742, 'a3': 0.231, 'a6': 0.897, 'a8': 0.519, 'a7': 0.849}

        self.assertEqual(initial_strengths, expected_initial_strengths, f"Expected initial strengths: {expected_initial_strengths}, but got: {initial_strengths}")

            
    def test_convergence_single_file(self):
        """
        Test the convergence of a single file.
        """
        # ─── Config ──────────────────────────────────────────────────────────────
        INPUT_DIR       = Path("data_generation/abaf/").resolve()
        MAX_FILES       = 0       # 0 = no limit
        MIN_SENTENCES   = 0
        MAX_SENTENCES   = 100
        BASE_SCORES     = "random"
        MAX_STEPS       = 20
        EPSILON         = 1e-3
        DELTA           = 5
        
        pattern_s = re.compile(r"_s(\d+)_")
        all_aba   = sorted(INPUT_DIR.glob("*.aba"))
        aba_paths = [
            p for p in all_aba
            if (m := pattern_s.search(p.name))
            and MIN_SENTENCES <= int(m.group(1)) <= MAX_SENTENCES
        ]
        if MAX_FILES > 0:
            aba_paths = aba_paths[:MAX_FILES]

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

        aba_path = "data_generation/abaf/nf_atm_s20_n0.01_a0.5_r2_b16_0.aba"

        # 1) load & build BSAF (heavy!)
        abaf = ABAF(path=str(aba_path))
        print(f"Loaded {aba_path.split('/')[-1]}. Flat: {not abaf.non_flat}")
        bsaf = abaf.to_bsaf()  # this sets abaf.non_flat, but we trust disk flag

        # 4) for each model configuration
        for model_name, cfg in RUNS:

            # build & solve
            model       = DiscreteModular(
                            BSAF=bsaf,
                            aggregation=cfg["aggregation"],
                            influence=cfg["influence"],
                            set_aggregation=cfg["set_aggregation"]
                        )
            final_state = model.solve(20, generate_plot=True, verbose=False)

            # record final strengths & convergence
            per_arg         = model.has_converged(epsilon=EPSILON, last_n=DELTA)
            global_conv     = model.is_globally_converged(epsilon=EPSILON, last_n=DELTA)
            conv_time       = model.convergence_time(epsilon=EPSILON, consecutive=DELTA, out_mean=True)
            total           = len(per_arg)
            prop_conv       = (sum(per_arg.values()) / total) if total else 0.0
        
            print(f"Model: {model_name}, Global Convergence: {global_conv}, Proportion Converged: {prop_conv}, Convergence Time: {conv_time}")


        self.assertTrue(global_conv, f"Expected global convergence for {model_name}, but it was not converged.")

    def test_loading(self):
        """Test loading of BSAF from pickle.
        """
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
            cache_file = CACHE_DIR / (aba_path.stem + f"{BASE_SCORES}.bsaf.pkl")

            Argument.reset_identifiers()
            Assumption.reset_identifiers()

            # if cache exists and we're not overriding, just load
            if cache_file.exists() and not CACHE_OVERRIDE:
                print(f"[CACHE HIT]   {aba_path.name}", flush=True)
                with open(cache_file, "rb") as f:
                    return pickle.load(f)

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
        
        # ─── Config ──────────────────────────────────────────────────────────────
        INPUT_DIR       = Path("data_generation/abaf/").resolve()
        OUTPUT_PKL      = Path("convergence_results_to10m_nf_atm_e2_d5_s500_randinit_prod.pkl")
        CACHE_DIR       = Path(INPUT_DIR,"bsaf_frameworks")
        CACHE_OVERRIDE  = False # set to True to override existing cache files
        RESULT_OVERRIDE = False # set to True to override existing results

        SEED            = 42      # random seed for reproducibility
        MAX_FILES       = 0       # 0 = no limit
        MIN_SENTENCES   = 0
        MAX_SENTENCES   = 100
        TIMEOUT_SECONDS = 600     # per‐file timeout
        EPSILON         = 1e-3    # convergence epsilon
        DELTA           = 5       # convergence delta
        MAX_STEPS       = 200      # max steps for convergence
        BASE_SCORES     = '' # 'random' or '' (empty==DEFAULT_WEIGHTS)
        SET_AGGREGATION = SetProductAggregation() # SetProductAggregation() or SetMinAggregation()

        RUNS = [
            ("DF-QuAD", dict(
                aggregation     = ProductAggregation(),
                influence       = LinearInfluence(conservativeness=1),
                set_aggregation = SET_AGGREGATION
            )),
            ("QE", dict(
                aggregation     = SumAggregation(),
                influence       = QuadraticMaximumInfluence(conservativeness=1),
                set_aggregation = SET_AGGREGATION
            )),
        ]

        aba_path = Path("data_generation/abaf/nf_atm_s20_n0.01_a0.5_r2_b16_0.aba")
        is_non_flat = disk_non_flat(aba_path)

        param_pat = re.compile(
            r"_s(?P<s>\d+)_"
            r"n(?P<n>[\d.]+)_"
            r"a(?P<a>[\d.]+)_"
            r"r(?P<r>\d+)_"
            r"b(?P<b>\d+)"
        )

        m = param_pat.search(aba_path.name)
        params = (dict(
              s=int(m.group("s")),
              n=float(m.group("n")),
              a=float(m.group("a")),
              r=int(m.group("r")),
              b=int(m.group("b"))
        ) if m else dict(s=None,n=None,a=None,r=None,b=None))

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

        bag = abaf.to_bag(args=bsaf.arguments) 
        
        entries = []
        for model_name, cfg in RUNS:
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
            model       = DiscreteModular(
                            BSAF=bsaf,
                            aggregation=cfg["aggregation"],
                            influence=cfg["influence"],
                            set_aggregation=cfg["set_aggregation"]
                        )
            final_state = model.solve(MAX_STEPS, generate_plot=True, verbose=False)


class TestABAF(unittest.TestCase):

    def test_abaf_from_iccma_file(self):
        """Load an ICCMA file and convert it to ABAF format.
        # Example. The ABA framework with rules 
        # p :- q,a.
        # q :- .
        # r :- b,c. 
        # assumptions a,b,c
        # contraries a̅ = r, b̅ = s, c̅ = t 
        # is specified as follows, with atom-indexing a=1, b=2, c=3, p=4, q=5, r=6, s=7, t=8.

        p aba 8
        a 1
        a 2
        a 3
        c 1 6
        c 2 7
        c 3 8
        r 4 5 1
        r 5
        r 6 2 3        
        """
        
        iccma_file = f"{os.path.dirname(os.path.realpath(__file__))}/examples/ABA_ICCMA_input.iccma"
        # Assuming the file contains valid ICCMA format
        abaf = ABAF(path=iccma_file)
        # Check if the assumptions and rules are parsed correctly

        expected_assumptions = {'1', '2', '3'}
        expected_contraries = {'1': '6', '2': '7', '3': '8'}
        expected_rules = {
            '4': {'5', '1'},
            '5': set(),
            '6': {'2', '3'}
        }
        ## format assumtions and rules
        actual_assumptions = {assumption.name for assumption in abaf.assumptions}
        actual_contraries = {assumption.name: assumption.contrary for assumption in abaf.assumptions}
        actual_rules = {rule.head.name: set([b.name for b in rule.body]) for rule in abaf.rules}

        # Check assumptions
        self.assertEqual(actual_assumptions, expected_assumptions, f"Expected assumptions: {expected_assumptions}, but got: {set(abaf.assumptions)}")
        # Check contraries
        for assumption in abaf.assumptions:
            self.assertEqual(actual_contraries[assumption.name], expected_contraries[assumption.name], f"Expected contrary for {assumption.name}: {expected_contraries[assumption.name]}, but got: {assumption.contrary}")
        # Check rules
        for rule in abaf.rules:
            self.assertEqual(actual_rules[rule.head.name], expected_rules[rule.head.name], f"Expected rule for {rule.head.name}: {expected_rules[rule.head.name]}, but got: {actual_rules[rule.head.name]}")
        print("ABAF loaded from ICCMA file correctly.")

    def test_argument_compare(self):
        """Load an ICCMA file and convert it to ABAF format.
        # Example. The ABA framework with rules 
        # p :- q,a.
        # q :- .
        # r :- b,c. 
        # assumptions a,b,c
        # contraries a̅ = r, b̅ = s, c̅ = t 
        # is specified as follows, with atom-indexing a=1, b=2, c=3, p=4, q=5, r=6, s=7, t=8.

        p aba 8
        a 1
        a 2
        a 3
        c 1 6
        c 2 7
        c 3 8
        r 4 5 1
        r 5
        r 6 2 3        
        """
        
        iccma_file = "examples/ABA_ICCMA_input.iccma"
        # Assuming the file contains valid ICCMA format
        abaf = ABAF(path=iccma_file, arg_mode='prune_supersets')
        # Check if the assumptions and rules are parsed correctly

        ## Compare arguments to derived using cling
        abaf._build_arguments(SetProductAggregation())

        ### extract all the derived atoms from ASP and compare to the arguments
        args_claims = set()
        for atom in abaf.derived_in:
            args_claims.add(atom)
        
        ## build the arguments
        abaf.build_arguments_procedure(SetProductAggregation())

        ## Format the arguments
        args = set()
        for arg in abaf.arguments:
            args.add(arg.claim.name)

        # Check that abaf.derived == {abaf.arguments}
        self.assertEqual(args_claims, args, f"Expected derived arguments: {abaf.arguments}, but got: {abaf.derived_in}")
        print("Arguments derived procedurally and from ASP are the same.")


    def test_argument_compare2(self):
        
        for i in range(0, 4):
            iccma_file = f"../dependency-graph-alternative/input_data_nf/non_flat_1_s25_c0.02_n0.2_a0.3_r5_b5_{i}.aba"
            # Assuming the file contains valid ICCMA format
            abaf = ABAF(path=iccma_file, arg_mode='prune_supersets')
            # Check if the assumptions and rules are parsed correctly

            ## Compare arguments to derived using cling
            abaf._build_arguments(SetProductAggregation())

            ### extract all the derived atoms from ASP and compare to the arguments
            args_claims = set()
            for atom in abaf.derived_in:
                args_claims.add(atom)
            ## collect the premises from the ASP procedure
            # args_premises = []
            # for counter in range(1,abaf.arg_counter):
            #     args_premises.append(abaf.derives[counter])

            
            ## build the arguments
            abaf.build_arguments_procedure(SetProductAggregation())

            ## Format the arguments
            args_claims2 = set()
            for arg in abaf.arguments:
                args_claims2.add(arg.claim.name)
            ## collect the premises from the arguments
            # args_premises2 = []
            # for arg in abaf.arguments:
            #     args_premises2.append(arg.premise)
            ## check the premises of the arguments - see if they are the same

            # Check that abaf.derived == {abaf.arguments}
            self.assertEqual(args_claims, args_claims2, f"Expected derived arguments: {abaf.arguments}, but got: {abaf.derived_in}")
            ## Check that the premises are the same
            # self.assertEqual(args_premises, args_premises2, f"Expected derived arguments: {args_premises}, but got: {args_premises2}")
            print("Arguments derived procedurally and from ASP are the same.")



    def test_abaf_from_iccma_file2(self):

        """Load an ICCMA file and convert it to ABAF format.
            p aba 6

            a 1
            a 2
            a 3
            a 4
            a  5
            c 1 2
            c 2 1
            c 3 4
            c 4 3
            c 4 2

            c 2 4
            c 3 4
            c 5  1
            c  1 5

            r 6 2    3  5
            r 6 4
        """
        
        iccma_file = "examples/ex2.iccma"
        
        ### Catch that the ABAF errors because of multiple contraries
        try:
            abaf = ABAF(path=iccma_file)
        except Exception as e:
            print(f"ABAF cannot be built: {e}")
            self.assertTrue(True)
        else:
            self.assertTrue(False, "Expected an error when loading ABAF from ICCMA file with multiple contraries, but no error was raised.")

    def test_abaf_flatness(self):
        
        iccma_file = "data_generation/abaf/nf_atm_s20_n0.01_a0.5_r2_b16_0.aba"
        
        ### Catch that the ABAF errors because of multiple contraries
        abaf = ABAF(path=iccma_file)
        print(f"ABAF non flat: {abaf.non_flat}")
        abaf.build_arguments_procedure(SetProductAggregation())

        print(abaf)
        self.assertFalse(abaf.non_flat, "Expected the ABAF to be flat, but it was not.")
        
        ### Non flat example
        iccma_file = "data_generation/abaf/nf_atm_s20_n0.1_a0.5_r2_b16_0.aba"
        
        ## Catch that the ABAF errors because of multiple contraries
        abaf = ABAF(path=iccma_file)
        print(f"ABAF non flat: {abaf.non_flat}")
        abaf.build_arguments_procedure(SetProductAggregation())

        print(abaf)

        self.assertTrue(abaf.non_flat, "Expected the ABAF to be non-flat, but it was not.")        

    # def argument_creation_speed_test(self):

    #     procedure1_times = []
    #     procedure2_times = []

    #     for i in range(0, 4):
    #         iccma_file = f"../dependency-graph-alternative/input_data_nf/non_flat_1_s25_c0.02_n0.2_a0.3_r5_b5_{i}.aba"            
    #         ## Compare arguments to derived using cling
    #         ## Sped up prcedure
    #         abaf = ABAF(path=iccma_file)
    #         start = time.time()
    #         args = abaf._build_arguments(SetProductAggregation())
    #         print("Time to build arguments: ", time.time() - start)
    #         ## collect times
    #         procedure1_times.append(time.time() - start)

    #         abaf1 = ABAF(path=iccma_file)
    #         ## Original prcedure
    #         start = time.time()
    #         args1 = abaf1.build_arguments_procedure_og(SetProductAggregation())
    #         print("Time to build arguments: ", time.time() - start)
    #         ## collect times
    #         procedure2_times.append(time.time() - start)
    #         ## Check that the arguments are the same
    #         # self.assertEqual(args, args1, f"Expected derived arguments: {args}, but got: {args1}")

    #         ## print average and std
    #         print(f"Procedure 1 avg Time (std): {sum(procedure1_times)/len(procedure1_times)} ({math.sqrt(sum([(x - sum(procedure1_times)/len(procedure1_times))**2 for x in procedure1_times])/len(procedure1_times))})")
    #         print(f"Procedure 2 avg Time (std): {sum(procedure2_times)/len(procedure2_times)} ({math.sqrt(sum([(x - sum(procedure2_times)/len(procedure2_times))**2 for x in procedure2_times])/len(procedure2_times))})")



    def rerun_unfinished_flat(self):
        """
        Rerun unfinished flat files.
        """

        # 2) Load your previously‐computed runs
        with open("convergence_results_to10m_nf_atm.pkl","rb") as pf:
            runs = pickle.load(pf)

        # 3) Group timeouts by file
        file_timeouts = defaultdict(list)
        for r in runs:
            file_timeouts[r["file"]].append(r.get("timeout"))

        # ─── Config ──────────────────────────────────────────────────────────────
        INPUT_DIR       = Path("data_generation/abaf/").resolve()
        OUTPUT_PKL      = "convergence_results_to10m_nf_atm.pkl"
        MAX_FILES       = 0       # 0 = no limit
        MIN_SENTENCES   = 0
        MAX_SENTENCES   = 100
        TIMEOUT_SECONDS = 600      # per‐file timeout

        pattern_s = re.compile(r"_s(\d+)_")
        all_aba   = sorted(INPUT_DIR.glob("*.aba"))
        aba_paths = [
            p for p in all_aba
            if (m := pattern_s.search(p.name))
            and MIN_SENTENCES <= int(m.group(1)) <= MAX_SENTENCES
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
        count = 0
        for aba_path, timeout in file_timeouts.items():

            # aba_path = "data_generation/abaf/nf_atm_s20_n0.01_a0.5_r2_b16_0.aba"
            if any(t==False for t in timeout):
                print(f"Skipping {aba_path} with timeout {timeout}")
                continue

            count += 1
            print(f"{count}: Processing {aba_path} with timeout {timeout}")
            # 1) load & build BSAF (heavy!)
            abaf = ABAF(path=Path(INPUT_DIR,aba_path))
            print(f"Loaded {aba_path.split('/')[-1]}. Flat: {not abaf.non_flat}")
            bsaf = abaf.to_bsaf()  # this sets abaf.non_flat, but we trust disk flag



            # 4) for each model configuration
            for model_name, cfg in RUNS:

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
            
                print(f"Model: {model_name}, Global Convergence: {global_conv}, Proportion Converged: {prop_conv}")

class TestBAG(unittest.TestCase):

    def test_convergence_single_file_flat(self):
        """
        Test the convergence of a single file flat.
        """
        # ─── Config ──────────────────────────────────────────────────────────────
        INPUT_DIR       = Path("data_generation/abaf/").resolve()
        MAX_FILES       = 0       # 0 = no limit
        MIN_SENTENCES   = 0
        MAX_SENTENCES   = 100
        BASE_SCORES     = "random"
        MAX_STEPS       = 20
        EPSILON         = 1e-3
        DELTA           = 5
        
        pattern_s = re.compile(r"_s(\d+)_")
        all_aba   = sorted(INPUT_DIR.glob("*.aba"))
        aba_paths = [
            p for p in all_aba
            if (m := pattern_s.search(p.name))
            and MIN_SENTENCES <= int(m.group(1)) <= MAX_SENTENCES
        ]
        if MAX_FILES > 0:
            aba_paths = aba_paths[:MAX_FILES]

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

        aba_path = "data_generation/abaf/nf_atm_s20_n0.01_a0.5_r2_b16_0.aba"

        abaf = ABAF(path=str(aba_path))
        print(f"Loaded {aba_path.split('/')[-1]}. Flat: {not abaf.non_flat}")
        bag = abaf.to_bag()  # this sets abaf.non_flat, but we trust disk flag
        print(bag)
        # 4) for each model configuration
        for model_name, cfg in RUNS:

            # build & solve
            model       = DiscreteModularBAG(
                            bag=bag,
                            aggregation=cfg["aggregation"],
                            influence=cfg["influence"],
                            set_aggregation=cfg["set_aggregation"]
                        )

            # record final strengths & convergence
            final_state = model.solve(20, generate_plot=True, verbose=False) ## this is the final state of the model at argument level

            ### extract final strengths for assumptions
            final_strengths = model.get_assumptions_strengths(set(abaf.assumptions))


            # record final strengths & convergence
            per_arg         = model.has_converged(epsilon=EPSILON, last_n=DELTA)
            global_conv     = model.is_globally_converged(epsilon=EPSILON, last_n=DELTA)
            conv_time       = model.convergence_time(epsilon=EPSILON, consecutive=DELTA, out_mean=True)
            total           = len(per_arg)
            prop_conv       = (sum(per_arg.values()) / total) if total else 0.0
        
            print(f"Model: {model_name}, Global Convergence: {global_conv}, Proportion Converged: {prop_conv}, Convergence Time: {conv_time}")

        self.assertTrue(global_conv, f"Expected global convergence for {model_name}, but it was not converged.")

    def test_convergence_single_file_nonflat(self):

        """
        Test the convergence of a single file non-flat.
        """
        # ─── Config ──────────────────────────────────────────────────────────────
        INPUT_DIR       = Path("data_generation/abaf/").resolve()
        MAX_FILES       = 0       # 0 = no limit
        MIN_SENTENCES   = 0
        MAX_SENTENCES   = 100
        BASE_SCORES     = "random"
        MAX_STEPS       = 20
        EPSILON         = 1e-3
        DELTA           = 5
        
        pattern_s = re.compile(r"_s(\d+)_")
        all_aba   = sorted(INPUT_DIR.glob("*.aba"))
        aba_paths = [
            p for p in all_aba
            if (m := pattern_s.search(p.name))
            and MIN_SENTENCES <= int(m.group(1)) <= MAX_SENTENCES
        ]
        if MAX_FILES > 0:
            aba_paths = aba_paths[:MAX_FILES]

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

        aba_path = "data_generation/abaf/nf_atm_s20_n0.1_a0.5_r2_b16_0.aba"

        # 1) load & build BSAF (heavy!)
        abaf = ABAF(path=str(aba_path))
        print(f"Loaded {aba_path.split('/')[-1]}. Flat: {not abaf.non_flat}")
        bag = abaf.to_bag()
        # 4) for each model configuration
        for model_name, cfg in RUNS:

            # build & solve
            model       = DiscreteModularBAG(
                            bag=bag,
                            aggregation=cfg["aggregation"],
                            influence=cfg["influence"],
                            set_aggregation=cfg["set_aggregation"]
                        )

            # record final strengths & convergence
            final_state = model.solve(20, generate_plot=True, verbose=False)
            ### extract final strengths for assumptions
            asm_stregths, agg_asm_strenghts = model.get_assumptions_strengths(set(abaf.assumptions), SetProductAggregation())
            # record final strengths & convergence
            per_arg         = model.has_converged(epsilon=EPSILON, last_n=DELTA)
            global_conv     = model.is_globally_converged(epsilon=EPSILON, last_n=DELTA)
            conv_time       = model.convergence_time(epsilon=EPSILON, consecutive=DELTA, out_mean=True)
            total           = len(per_arg)
            prop_conv       = (sum(per_arg.values()) / total) if total else 0.0
            print(f"Model: {model_name}, Global Convergence: {global_conv}, Proportion Converged: {prop_conv}, Convergence Time: {conv_time}")
        self.assertTrue(global_conv, f"Expected global convergence for {model_name}, but it was not converged.")



    def test_convergence_single_file_nonflat_asm_view(self):

        """
        Test the convergence of a single file non-flat.
        """
        # ─── Config ──────────────────────────────────────────────────────────────
        INPUT_DIR       = Path("data_generation/abaf/").resolve()
        MAX_FILES       = 0       # 0 = no limit
        MIN_SENTENCES   = 0
        MAX_SENTENCES   = 100
        BASE_SCORES     = "random"
        MAX_STEPS       = 20
        EPSILON         = 1e-3
        DELTA           = 5
        
        pattern_s = re.compile(r"_s(\d+)_")
        all_aba   = sorted(INPUT_DIR.glob("*.aba"))
        aba_paths = [
            p for p in all_aba
            if (m := pattern_s.search(p.name))
            and MIN_SENTENCES <= int(m.group(1)) <= MAX_SENTENCES
        ]
        if MAX_FILES > 0:
            aba_paths = aba_paths[:MAX_FILES]

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

        aba_path = "data_generation/abaf/nf_atm_s20_n0.01_a0.5_r2_b16_0.aba"

        # 1) load & build BSAF (heavy!)
        abaf = ABAF(path=str(aba_path))
        print(f"Loaded {aba_path.split('/')[-1]}. Flat: {not abaf.non_flat}")
        bag = abaf.to_bag()
        # 4) for each model configuration
        for model_name, cfg in RUNS:

            # build & solve
            model       = DiscreteModularBAG(
                            bag=bag,
                            aggregation=cfg["aggregation"],
                            influence=cfg["influence"],
                            set_aggregation=cfg["set_aggregation"]
                        )

            # record final strengths & convergence
            final_state = model.solve(20, generate_plot=True, verbose=False, view='Assumptions', 
                                      assumptions=abaf.assumptions, aggregate_strength_f=SetProductAggregation())
            ### extract final strengths for assumptions
            asm_stregths, agg_asm_strenghts = model.get_assumptions_strengths(set(abaf.assumptions), SetProductAggregation())
            # record final strengths & convergence
            per_arg         = model.has_converged(epsilon=EPSILON, last_n=DELTA)
            global_conv     = model.is_globally_converged(epsilon=EPSILON, last_n=DELTA)
            conv_time       = model.convergence_time(epsilon=EPSILON, consecutive=DELTA, out_mean=True)
            total           = len(per_arg)
            prop_conv       = (sum(per_arg.values()) / total) if total else 0.0
            print(f"Model: {model_name}, Global Convergence: {global_conv}, Proportion Converged: {prop_conv}, Convergence Time: {conv_time}")
        self.assertTrue(global_conv, f"Expected global convergence for {model_name}, but it was not converged.")


#------------------------ RUN TESTS ------------------------#
# TestABAF().test_abaf_from_iccma_file()
# TestABAF().test_abaf_from_iccma_file2()
# TestABAF().test_argument_compare()
# TestABAF().test_argument_compare2()
# TestABAF().test_abaf_flatness()       
# TestABAF().argument_creation_speed_test()
# TestABAF().rerun_unfinished_flat()

# TestBSAF().test_aba_bsaf_1()
# TestBSAF().test_aba_bsaf_2()

TestBSAF().test_convergence_single_file()
TestBSAF().test_convergence_random_weight()
TestBSAF().test_loading()

TestBAG().test_convergence_single_file_flat()
TestBAG().test_convergence_single_file_nonflat()
TestBAG().test_convergence_single_file_nonflat_asm_view()

print("All tests passed.")

