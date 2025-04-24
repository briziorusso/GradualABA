import matplotlib.pyplot as plt

from ABAF.Rule import Rule
from ABAF.Assumption import Assumption
from ABAF.Sentence import Sentence
from ABAF.ABAF import ABAF
from ABAF.parser.asp_parser import ASPParser
from ABAF.DependencyGraph import DependencyGraph
from constants import DEFAULT_WEIGHT

# Define sentences
p = Sentence("p")
q = Sentence("q")
r = Sentence("r")
s = Sentence("s")
t = Sentence("t")

# Define assumptions
u = Assumption("u", initial_weight=DEFAULT_WEIGHT)
v = Assumption("v", initial_weight=DEFAULT_WEIGHT)
asn2 = Assumption("v", contrary="p", initial_weight=DEFAULT_WEIGHT)

print("This is the weight of u: ", u.initial_weight, "\n\n")

# Define rules
rule1 = Rule(head=p, name="rule1", body=[q, r])
print(rule1)  # rule1: p :- q, r.

rule2 = Rule(head=s, name="rule2")
print(rule2)  # rule2: s.

rule3 = Rule(head=t, name="rule3", body=[p])
print(rule3)  # rule3: t :- p.

rule4 = Rule(head=u, name="rule4", body=[p, q])
print(rule4)  # rule4: u :- p, q.

# Input string for ASP parser
input_str = """
assumption(a).
assumption(b).

contrary(a, p).
contrary(b,x).

head(1, p).
body(1,q).
body(1,a ).

head(2, p ).
body(2,b).

head(3,x).
"""

# Parse the input string
abafnn = ASPParser.parse(input_str)

# Add assumptions and rules to the ABAF instance
abafnn.add_assumption(u)
abafnn.add_assumption(asn2)
abafnn.add_rules([rule1, rule2, rule3, rule4])

# Print the ABAF object
print(abafnn)
print(repr(abafnn))

# Directly access and print the assumptions
assumptions = abafnn.assumptions
print("Assumptions:", assumptions)
for assumption in assumptions:
    print(assumption.name, assumption.contrary, assumption.initial_weight)

# Collect and print the sentences
sentences = abafnn.collect_sentences()
print("Sentences:", sentences)
for sentence in sentences:
    print(sentence.name, sentence.initial_weight)

# Create and visualize the dependency graph
dep_graph = DependencyGraph()
dep_graph.construct_from_abaf(abafnn)
print(dep_graph)

# Update node weights
dep_graph.update_node_weight("a", 0.8)
dep_graph.update_nodes_weights({"q": 0.6, "r": 0.7, "rule1": 0.9})
print("Updated graph:")
print(dep_graph)

# Update ABAF weights based on the graph
# dep_graph.update_abaf_weights(abafnn) ## <-- This errors because some nodes are empty (v)
print("Updated ABAF:")
print(abafnn)

# dep_graph.draw()

# Convert to BAG
bag = abafnn.to_bag()
print("\n=== Converted BAG ===")
print(bag)             # uses BAG.__str__()
print(repr(bag))       # full repr with arguments, attacks & supports

# Convert to BSAF
# Note: to_bsaf needs the same set of assumptions you passed into the constructor
bsaf = abafnn.to_bsaf()
print("\n=== Converted BSAF ===")
print(bsaf)            # uses BSAF.__str__()
print(repr(bsaf))      # full repr with sets of attacker/supporter coalitions

# (Optionally) Inspect the raw structures:
print("\nBAG internal sets:")
print(" Arguments:", {a for a in bag.arguments})
print(" Attacks:  ", { (att.attacker.name, att.attacked.name) for att in bag.attacks })
print(" Supports: ", { (sup.supporter.name, sup.supported.name) for sup in bag.supports })

print("\nBSAF internal maps:")
for asm in bsaf.assumptions:
    print("Assumption:", asm.name)
    print("   attacks:", bsaf.attacks[asm])
    print("   supports:", bsaf.supports[asm])



# # ------------------ Unit Tests ------------------
# if __name__ == "__main__":
#     import unittest

#     class DummyAssumption(Assumption):
#         def __init__(self, name, contrary=None, initial_weight=1):
#             self.name = name
#             self.contrary = contrary
#             self.initial_weight = weight

#     class DummySentence(Sentence):
#         def __init__(self, name):
#             self.name = name

#     class RuleStub(Rule):
#         def __init__(self, head, body=None, name=None):
#             self.head = head
#             self.body = body or []
#             self.name = name

#     class TestABAFConversions(unittest.TestCase):
#         def setUp(self):
#             a = DummyAssumption('a', contrary='b', initial_weight=1)
#             b = DummyAssumption('b', contrary='a', initial_weight=2)
#             self.abaf = ABAF(assumptions=[a, b])
#             c = DummySentence('c')
#             self.abaf.rules = []
#             self.abaf.rules.append(RuleStub(head=c, body=[DummySentence('a'), DummySentence('b')], name='r1'))

#         def test_to_bsaf(self):
#             bsaf = self.abaf.to_bsaf()
#             names = {arg.name for arg in bsaf.arguments}
#             self.assertSetEqual(names, {'a', 'b', 'r1'})
#             # Support: r1 supported by a and b
#             sup_sets = bsaf.setSupports[next(arg for arg in bsaf.arguments if arg.name=='r1')]
#             self.assertIn(
#                 sorted([x.name for x in sup_sets[0]]),
#                 [['a', 'b']]
#             )
#             # No attacks between composite args
#             att_sets = bsaf.setAttacks
#             self.assertTrue(all(len(v)==0 for v in att_sets.values()))

#         def test_to_bag(self):
#             bag = self.abaf.to_bag()
#             self.assertSetEqual(set(bag.arguments.keys()), {'a', 'b', 'r1'})
#             supports = [(sup.supporter.name, sup.supported.name) for sup in bag.supports]
#             self.assertIn(('a', 'r1'), supports)
#             self.assertIn(('b', 'r1'), supports)
#             self.assertEqual(len(bag.attacks), 0)

#     unittest.main(verbosity=2)
