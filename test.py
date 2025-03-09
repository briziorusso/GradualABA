from Rule import Rule
from ABAF import ABAF

# Rule 1 with custom name and body
rule1 = Rule(head="p", name="CustomRule1", body={"q", "r"})
print(rule1)  # CustomRule1: p ← q, r

# Rule 2 with default name (Rule2) and no body
rule2 = Rule(head="s")
print(rule2)  # Rule2: s

# Rule 3 with default name (Rule3) and a custom body
rule3 = Rule(head="t", body={"p"})
print(rule3)  # Rule3: t ← p

# Rule with no name provided will automatically generate Rule4
rule4 = Rule(head="u", body={"p", "q"})
print(rule4)  # Rule4: u ← p, q

def test_abaf_framework():
    # Create an ABAF instance with no assumptions or rules
    abaf = ABAF()

    # Add assumptions and contraries
    abaf.add_assumption("p")
    abaf.add_assumption("q", "¬q")

    # Add custom rules
    abaf.add_rule(head="p", body={"q", "r"})
    abaf.add_rule(head="s", body={"t"})
    abaf.add_rule(head="t", name=3)
    
    # Collect all sentences (atoms, assumptions, contraries, rules' heads/ bodies)
    sentences = abaf.collect_sentences()
    print("Collected Sentences:", sentences)
    
    # Print ABAF object using __str__()
    print(abaf)

    # Print ABAF object using __repr__()
    print(repr(abaf))

if __name__ == "__main__":
    test_abaf_framework()

from parser.asp_parser import ASPParser

input_str = """
assumption(a).
assumption(b).

contrary(a, p).
contrary(b,x).

head(1, p).
body(1,q).

head(2, p ).
body(2,b).

head(3,x).
"""

# Parse the input string
abafnn = ASPParser.parse(input_str)

# Print the ABAF object
print(abafnn)
print(repr(abafnn))