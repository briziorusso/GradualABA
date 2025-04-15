import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants import DEFAULT_WEIGHT


class Argument:
    _counter = 0
    _existing_arguments = {}

    def __new__(cls, name=None, *args, **kwargs):
        # Generate a unique name if none is provided
        if name is None:
            name = f"x{cls._counter + 1}"
            cls._counter += 1

        # Check if an Argument with the same name already exists
        if name in cls._existing_arguments:
            raise ValueError(f"An Argument with the name '{name}' already exists.")
        #   return cls._existing_arguments[name]  # Return the existing instance


        # Create a new instance and store it in the dictionary
        instance = super(Argument, cls).__new__(cls)
        cls._existing_arguments[name] = instance
        instance.name = name  # Assign the name to the instance here
        return instance

    def __init__(self, name=None, initial_weight=DEFAULT_WEIGHT, strength=None):
        # Ensure __init__ is called only once per instance
        if not hasattr(self, 'initialized'):
            self.initial_weight = initial_weight
            self.strength = strength if strength is not None else initial_weight

            if not isinstance(initial_weight, (int, float)):
                raise TypeError("initial_weight must be of type integer or float")

            self.initialized = True

    def get_name(self):
        return self.name

    def get_initial_weight(self):
        return self.initial_weight

    def reset_initial_weight(self, weight):
        if not isinstance(weight, (int, float)):
            raise TypeError("weight must be of type integer or float")
        self.initial_weight = weight

    def __repr__(self) -> str:
        return f"Argument(name={self.name}, weight={self.initial_weight}, strength={self.strength})"

    def __str__(self) -> str:
        return self.__repr__()
