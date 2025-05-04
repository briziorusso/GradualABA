from .Sentence import Sentence
import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import DEFAULT_WEIGHT


class Assumption(Sentence):
    def __new__(cls, name, contrary=None, initial_weight=DEFAULT_WEIGHT):
        if name in Sentence._existing_sentences:
            raise ValueError(f"A sentence with the name '{name}' already exists.")
        instance = super().__new__(cls)
        return instance

    def __init__(self, name, contrary=None, initial_weight=DEFAULT_WEIGHT):
        if not hasattr(self, 'initialized'):  # Ensure __init__ is called only once
            super().__init__(name, initial_weight)
            self.contrary = contrary
            self.initialized = True

    def __reduce__(self):
        """
        Tell pickle how to rebuild me:
         - the first element is the callable (here the class itself)
         - the second is the tuple of arguments to pass to __new__/__init__
        """
        return (
            Assumption,                          # callable
            (self.name, self.contrary, self.initial_weight)
        )

    def __str__(self):
        return f"Assumption({self.name}, contrary={self.contrary}, weight={self.initial_weight})"

    def __repr__(self):
        return self.__str__()
    
    def __contrary__(self):
        return self.contrary
    
    def __eq__(self, other):
        if not isinstance(other, Assumption):
            return False
        return self.name == other.name and self.contrary == other.contrary and self.initial_weight == other.initial_weight
    
    def __hash__(self):
        return hash((self.name, self.contrary))
    
    @classmethod
    def reset_identifiers(cls):
        """Reset the used identifiers (for testing or reloading purposes)."""
        cls._existing_sentences.clear()
    