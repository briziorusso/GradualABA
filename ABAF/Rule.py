from ABAF.Sentence import Sentence
import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import DEFAULT_WEIGHT

class Rule:
    _used_identifiers = set()
    _counter = 1

    def __init__(self, head: Sentence, body=None, name=None):
        if not head:
            raise ValueError("Head must be specified.")

        if name is None:
            name = f"r{Rule._counter}"
            Rule._counter += 1

        if name in Rule._used_identifiers:
            raise ValueError(f"Rule identifier '{name}' already exists. Choose a unique name.")

        self.name = name
        self.head = head
        self.body = body if body else []
        self.weight = DEFAULT_WEIGHT  # Use the default weight

        # Register the identifier as used
        Rule._used_identifiers.add(name)

    def __repr__(self):
        body_str = ", ".join([str(sentence.name) for sentence in self.body]) if self.body else ""
        if body_str:
            return f"{self.name}[{self.weight}]: {self.head.name} :- {body_str}."
        else:
            return f"{self.name}[{self.weight}]: {self.head.name}."

    def update_weight(self, new_weight):
        if not isinstance(new_weight, (int, float)):
            raise TypeError("new_weight must be of type integer or float")
        self.weight = new_weight

    @classmethod
    def reset_identifiers(cls):
        """Reset the used identifiers (for testing or reloading purposes)."""
        cls._used_identifiers.clear()
        cls._counter = 1  # Reset counter for fresh rule names
