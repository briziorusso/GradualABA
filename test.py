import matplotlib.pyplot as plt

from Rule import Rule
from Assumption import Assumption
from Sentence import Sentence
from ABAF import ABAF
from parser.asp_parser import ASPParser
from DependencyGraph import DependencyGraph
from constants import DEFAULT_WEIGHT

# Define sentences
p = Sentence("p")
q = Sentence("q")
r = Sentence("r")
s = Sentence("s")
t = Sentence("t")

# Define assumptions
u = Assumption("u", weight=DEFAULT_WEIGHT)
asn2 = Assumption("v", contrary="p", weight=DEFAULT_WEIGHT)

print("This is the weight of u: ", u.weight, "\n\n")

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
    print(assumption.name, assumption.contrary, assumption.weight)

# Collect and print the sentences
sentences = abafnn.collect_sentences()
print("Sentences:", sentences)
for sentence in sentences:
    print(sentence.name, sentence.weight)

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
dep_graph.update_abaf_weights(abafnn)
print("Updated ABAF:")
print(abafnn)

# dep_graph.draw()