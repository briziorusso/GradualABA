import re
from .. import Rule
from .. import Assumption
from .. import Sentence
from .. import ABAF

class ASPParser:

    @staticmethod
    def parse(input_str):
        assumptions = []  # List of Assumption objects
        rules = []

        # Regular expressions to match different parts of the input
        assumption_pattern = r"assumption\(\s*(\w+)\s*\)."
        contrary_pattern = r"contrary\(\s*(\w+)\s*,\s*(\w+)\s*\)."
        head_pattern = r"head\(\s*(\d+)\s*,\s*(\w+)\s*\)."
        body_pattern = r"body\(\s*(\d+)\s*,\s*(\w+)\s*\)."
        
        # Temporary dictionary to hold assumptions and their contraries
        temp_assumptions = {}

        # Parsing assumptions
        for match in re.finditer(assumption_pattern, input_str):
            temp_assumptions[match.group(1)] = None  # Initialize assumption with no contrary

        # Parsing contraries
        for match in re.finditer(contrary_pattern, input_str):
            assumption = match.group(1)
            contrary = match.group(2)
            if assumption in temp_assumptions:
                temp_assumptions[assumption] = contrary  # Assign contrary to assumption
            else:
                raise ValueError(f"Contrary for unknown assumption: {assumption}")

        # Create Assumption objects
        for name, contrary in temp_assumptions.items():
            assumptions.append(Assumption(name, contrary))

        # Dictionaries for heads and bodies
        rule_heads = {}
        rule_bodies = {}

        # Parsing rule heads
        for match in re.finditer(head_pattern, input_str):
            rule_id = int(match.group(1))
            head = Sentence(match.group(2))
            rule_heads[rule_id] = head  # Add head to the dictionary
        
        # Parsing rule bodies and adding them to the body dictionary
        for match in re.finditer(body_pattern, input_str):
            rule_id = int(match.group(1))
            body = Sentence(match.group(2))
            if rule_id not in rule_bodies:
                rule_bodies[rule_id] = []  # Initialize the list for the body
            rule_bodies[rule_id].append(body)  # Add body atom to the list

        # Now, construct the Rule objects from the dictionaries
        for rule_id in rule_heads:
            head = rule_heads[rule_id]
            body = rule_bodies.get(rule_id, [])  # Get the body, default to empty list if none
            rule = Rule(
                head=head,
                body=body,
                name=f"pr{rule_id}"
            )
            rules.append(rule)
        
        # Return an ABAF object
        return ABAF(assumptions=assumptions, rules=rules)
