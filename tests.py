import unittest

from BSAF.BSAF import BSAF
from BSAF.Argument import Argument
from ABAF.Assumption import Assumption
from ABAF.Rule import Rule
from ABAF.Sentence import Sentence
from ABAF.ABAF import ABAF

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
        
        iccma_file = "examples/ABA_ICCMA_input.iccma"
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
        actual_rules = {rule.head.name: set(rule.body) for rule in abaf.rules}

        # Check assumptions
        self.assertEqual(actual_assumptions, expected_assumptions, f"Expected assumptions: {expected_assumptions}, but got: {set(abaf.assumptions)}")
        # Check contraries
        for assumption in abaf.assumptions:
            self.assertEqual(actual_contraries[assumption.name], expected_contraries[assumption.name], f"Expected contrary for {assumption.name}: {expected_contraries[assumption.name]}, but got: {assumption.contrary}")
        # Check rules
        for rule in abaf.rules:
            self.assertEqual(actual_rules[rule.head.name], expected_rules[rule.head.name], f"Expected rule for {rule.head.name}: {expected_rules[rule.head.name]}, but got: {abaf.rules[rule.head.name]}")
        print("ABAF loaded from ICCMA file correctly.")


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
        # Assuming the file contains valid ICCMA format
        abaf = ABAF(path=iccma_file)
        # Check if the assumptions and rules are parsed correctly

        expected_assumptions = {'1', '2', '3', '4', '5'}
        expected_contraries = {
            '1': '2',
            '2': '1',
            '3': '4',
            '4': '3',
            '4': '2',
            '2': '4',
            '3': '4',
            '5': '1',
            '1': '5'
        }
        expected_rules = {
            '6': {'2', '3', '5'},
            '4': set()
        }
        ## format assumtions and rules
        actual_assumptions = {assumption.name for assumption in abaf.assumptions}
        actual_contraries = {assumption.name: assumption.contrary for assumption in abaf.assumptions}
        actual_rules = {rule.head.name: set(rule.body) for rule in abaf.rules}
        # Check assumptions
        self.assertEqual(actual_assumptions, expected_assumptions, f"Expected assumptions: {expected_assumptions}, but got: {set(abaf.assumptions)}")
        # Check contraries
        for assumption in abaf.assumptions:
            self.assertEqual(actual_contraries[assumption.name], expected_contraries[assumption.name], f"Expected contrary for {assumption.name}: {expected_contraries[assumption.name]}, but got: {assumption.contrary}")
        # Check rules
        for rule in abaf.rules:
            self.assertEqual(actual_rules[rule.head.name], expected_rules[rule.head.name], f"Expected rule for {rule.head.name}: {expected_rules[rule.head.name]}, but got: {abaf.rules[rule.head.name]}")
        print("ABAF loaded from ICCMA file correctly.")
        
        
TestBSAF().test_aba_bsaf_1()
TestBSAF().test_aba_bsaf_2()
TestABAF().test_abaf_from_iccma_file()
TestABAF().test_abaf_from_iccma_file2()

print("All tests passed.")

