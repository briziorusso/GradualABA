import re
from Rule import Rule  # Import Rule class from Rule.py
from ABAF import ABAF  # Import ABAF class from ABAF.py

class ASPParser:
    """Parser for the ASP-inspired ABA format."""
    
    @staticmethod
    def parse(input_str):
        assumptions = {}  # Dictionary of assumptions with contraries
        rules = []

        # Regular expressions to match different parts of the input
        assumption_pattern = r"assumption\(\s*(\w+)\s*\)."
        contrary_pattern = r"contrary\(\s*(\w+)\s*,\s*(\w+)\s*\)."
        head_pattern = r"head\(\s*(\d+)\s*,\s*(\w+)\s*\)."
        body_pattern = r"body\(\s*(\d+)\s*,\s*(\w+)\s*\)."
        
        # Parsing assumptions
        for match in re.finditer(assumption_pattern, input_str):
            assumptions[match.group(1)] = None  # Initialize assumption with no contrary


        # Parsing contraries
        for match in re.finditer(contrary_pattern, input_str):
            assumption = match.group(1)
            contrary = match.group(2)
            if assumption in assumptions:
                print("yay", assumption)
                assumptions[assumption] = contrary  # Assign contrary to assumption
            else:
                raise ValueError(f"Contrary for unknown assumption: {assumption}")

        # Dictionaries for heads and bodies
        rule_heads = {}
        rule_bodies = {}

        # Parsing rule heads
        for match in re.finditer(head_pattern, input_str):
            rule_id = int(match.group(1))
            head = match.group(2)
            rule_heads[rule_id] = head  # Add head to the dictionary
        
        # Parsing rule bodies and adding them to the body dictionary
        for match in re.finditer(body_pattern, input_str):
            rule_id = int(match.group(1))
            body = match.group(2)
            if rule_id not in rule_bodies:
                rule_bodies[rule_id] = set()  # Initialize the set for the body
            rule_bodies[rule_id].add(body)  # Add body atom to the set

        # Now, construct the Rule objects from the dictionaries
        for rule_id in rule_heads:
            head = rule_heads[rule_id]
            body = rule_bodies.get(rule_id, set())  # Get the body, default to empty set if none
            rule = Rule(
                head=head,
                body=body,
                name=f"Rule{rule_id}"
            )
            
            rules.append(rule)
        
        # Return an ABAF object
        return ABAF(assumptions=assumptions, rules=rules)
