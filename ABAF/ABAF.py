## TODO: functions for converting ABAF to BSAF and BAG (BAG means BAF)

from .Rule import Rule
from .Assumption import Assumption
from .Sentence import Sentence
from BAG import BAG

import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import DEFAULT_WEIGHT



class ABAF:
    def __init__(self, assumptions=None, rules=None):
        self.assumptions = assumptions if assumptions else []
        self.rules = rules if rules else []
        self.assumption_set = set(assumption.name for assumption in self.assumptions)

    def add_rule(self, head: Sentence, body=None, name=None):
        rule = Rule(head, body, name)
        self.rules.append(rule)

    def add_rules(self, rule_set):
        """Add multiple rules at once."""
        self.rules.extend(rule_set)

    def add_assumption(self, assumption: Assumption):
        """Add an assumption object."""
        self.assumption_set.add(assumption.name)
        self.assumptions.append(assumption)
    
    def add_assumptions(self, new_assumption_set):
        """Add multiple assumptions at once."""
        self.assumptions.extend(new_assumption_set) 
        self.assumption_set.update(assumption.name for assumption in new_assumption_set) 

    def collect_sentences(self):
        sentences = set() 

        # Add contraries and assumptions (avoid adding None)
        for assumption in self.assumptions:
            sentences.add(assumption)
            if assumption.contrary is not None:
                sentences.add(Sentence(assumption.contrary))

        # Add head and body of rules
        for rule in self.rules:
            sentences.add(rule.head)
            sentences.update(rule.body)

        return list(sentences)
    
    def to_bsaf(self):
        """ABAF into BSAF"""
        # TODO! 
        attacks = {}
        supports = {}
        return attacks, supports

    def to_bag(self):
        """ABAF into BAF (resp BAG)"""
        # TODO! 
        bag = BAG()
        return bag

    def __str__(self):
        contraries_str = "\n".join(f"-{assumption.name} = {assumption.contrary}" for assumption in self.assumptions)
        rules_str = "\n".join(map(str, self.rules))
        return f"\nAssumptions: {', '.join(f"{assumption.name}[{assumption.weight}]" for assumption in self.assumptions)}\n\nContraries:\n{contraries_str}\n\nRules:\n{rules_str}\n"

    def __repr__(self):
        return f"ABAF(Assumptions={self.assumptions}, Rules={self.rules})"
