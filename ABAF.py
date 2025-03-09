from Rule import Rule

class ABAF:
    """Represents an Assumption-Based Argumentation Framework (ABAF)."""

    def __init__(self, assumptions=None, rules=None):
        """
        assumptions: A set of assumptions (optional, default is empty).
        rules: A list of Rule objects (optional, default is empty).
        """
        self.assumptions = assumptions if assumptions else {}
        self.rules = rules if rules else []
        self.assumption_set = set(self.assumptions.keys())

    def add_rule(self, head, body=None, name=None):
        rule = Rule(head, body, name)
        self.rules.append(rule)

    def add_assumption(self, assumption, contrary=None):
        """Add an assumption and its contrary."""
        self.assumption_set.add(assumption)
        self.assumptions[assumption] = contrary

    def collect_sentences(self):
        """Collect all sentences (atoms, assumptions, contraries, and rule components)."""
        sentences = set(self.assumption_set) # Start with assumptions

        # Add contraries (avoid adding None)
        for contrary in self.assumptions.values():
            if contrary is not None:
                sentences.add(contrary)

        # Add head and body of rules
        for rule in self.rules:
            if rule.head:
                sentences.add(rule.head) 
            sentences.update(rule.body) 

        return sentences

    def __str__(self):
        """User-friendly string representation of the ABAF."""
        contraries_str = "\n".join(f"-{k} = {v}" for k, v in self.assumptions.items())
        rules_str = "\n".join(map(str, self.rules))
        return f"\nAssumptions: {",".join(map(str,self.assumption_set))}\n\nContraries:\n{contraries_str}\n\nRules:\n{rules_str}\n"

    def __repr__(self):
        """Technical string representation of the ABAF."""
        return f"ABAF(Assumptions={self.assumption_set}, Rules={self.rules}, Contraries={self.assumptions})"
