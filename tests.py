import unittest
import os, sys

from BSAF.BSAF import BSAF
from BSAF.Argument import Argument
from ABAF.Assumption import Assumption
from ABAF.Rule import Rule
from ABAF.Sentence import Sentence
from ABAF.ABAF import ABAF
from semantics.modular.SetProductAggregation import SetProductAggregation 

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
        self.assertEqual(args_claims, args, f"Expected derived arguments: {abaf.arguments}, but got: {abaf.derived}")
        print("Arguments derived procedurally and from ASP are the same.")


    def test_argument_compare2(self):
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
            self.assertEqual(args_claims, args_claims2, f"Expected derived arguments: {abaf.arguments}, but got: {abaf.derived}")
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


    ### attacks and supports correspont to arguments deriving assumptions and contraries of assumptions

    ### Check for FLAT and non-FLAT
        
        
TestBSAF().test_aba_bsaf_1()
TestBSAF().test_aba_bsaf_2()
TestABAF().test_abaf_from_iccma_file()
TestABAF().test_abaf_from_iccma_file2()
TestABAF().test_argument_compare()
TestABAF().test_argument_compare2()

print("All tests passed.")

