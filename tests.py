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
            arglist.add((frozenset(a.name for a in arg.body), arg.head.name))
            
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
            arglist.add((frozenset(a.name for a in arg.body), arg.head.name))

        
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
        
        
TestBSAF().test_aba_bsaf_1()
TestBSAF().test_aba_bsaf_2()

