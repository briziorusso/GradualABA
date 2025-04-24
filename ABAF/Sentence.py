import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import DEFAULT_WEIGHT

class Sentence:
    _counter = 0
    _existing_sentences = {}

    def __new__(cls, name=None, initial_weight=DEFAULT_WEIGHT):
        if name is None:
            name = f"Sentence{Sentence._counter + 1}"
            Sentence._counter += 1

        if name in Sentence._existing_sentences:
            return Sentence._existing_sentences[name]

        instance = super(Sentence, cls).__new__(cls)
        Sentence._existing_sentences[name] = instance
        return instance

    def __init__(self, name=None, initial_weight=DEFAULT_WEIGHT):
        if not hasattr(self, 'initialized'):  # Ensure __init__ is called only once
            if name is None:
                name = f"Sentence{Sentence._counter + 1}"
                Sentence._counter += 1

            self.name = name
            self.initial_weight = initial_weight

            if initial_weight is not None and not isinstance(initial_weight, (int, float)):
                raise TypeError("initial_weight must be of type integer or float")

            self.initialized = True

    def __str__(self):
        return f"Sentence({self.name}, initial_weight={self.initial_weight})"

    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        if not isinstance(other, Sentence):
            return False
        return self.name == other.name and self.initial_weight == other.initial_weight
    
    def __hash__(self):
        return hash((self.name, self.initial_weight))

    def update_weight(self, new_weight):
        if not isinstance(new_weight, (int, float)):
            raise TypeError("new_weight must be of type integer or float")
        self.initial_weight = new_weight