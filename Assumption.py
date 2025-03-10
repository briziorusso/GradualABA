from Sentence import Sentence
from constants import DEFAULT_WEIGHT

class Assumption(Sentence):
    def __new__(cls, name, contrary=None, weight=DEFAULT_WEIGHT):
        if name in Sentence._existing_sentences:
            raise ValueError(f"A sentence with the name '{name}' already exists.")
        instance = super(Assumption, cls).__new__(cls)
        return instance

    def __init__(self, name, contrary=None, weight=DEFAULT_WEIGHT):
        if not hasattr(self, 'initialized'):  # Ensure __init__ is called only once
            super().__init__(name, weight)
            self.contrary = contrary
            self.initialized = True

    def __str__(self):
        return f"Assumption({self.name}, contrary={self.contrary}, weight={self.weight})"

    def __repr__(self):
        return self.__str__()