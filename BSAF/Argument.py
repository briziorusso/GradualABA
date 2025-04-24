import sys
import os
from constants import DEFAULT_WEIGHT

# Ensure unique module resolution
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class Argument:
    """
    Unified Argument class combining:
      - Unique instance names (auto-generated or user-provided)
      - Initial weight, dynamic strength
      - Management of attackers and supporters with weights
      - Optional head (claim) and body (supporting assumptions) attributes
    """
    _counter = 0
    _instances = {}

    def __new__(cls, name=None, initial_weight=DEFAULT_WEIGHT, strength=None, head=None, body=None):
        # Generate unique name if not provided
        if name is None:
            name = f"x{cls._counter + 1}"
            cls._counter += 1
        # Prevent duplicate names
        if name in cls._instances:
            # raise ValueError(f"An Argument with the name '{name}' already exists.")
            ## Check that they are actually the same
            if cls._instances[name].head != head or cls._instances[name].body != body:
                # print(f"Instance with name '{name}' has different head or body. Generating new name.")
                name = f"x{cls._counter + 1}"
                cls._counter += 1
            else:
                # print(f"An Argument with the name '{name}' already exists. Using existing instance.")
                return cls._instances[name]
        # Create instance and store
        inst = super().__new__(cls)
        cls._instances[name] = inst
        inst.name = name
        return inst

    def __init__(self, name=None, initial_weight=DEFAULT_WEIGHT, strength=None, head=None, body=None):
        # Only initialize once
        if hasattr(self, '_initialized'):
            return
        # Validate weight
        if not isinstance(initial_weight, (int, float)):
            raise TypeError("initial_weight must be an int or float")
        self.initial_weight = initial_weight
        self.strength = strength if strength is not None else initial_weight
        # Optional head (Sentence or string representing the conclusion)
        self.head = head
        # Optional body: list of Sentence or Assumption instances supporting this argument
        self.body = list(body) if body is not None else []
        # Relationship maps: Argument -> numeric weight
        self.attackers = {}
        self.supporters = {}
        self._initialized = True

    def get_name(self):
        return self.name

    def get_initial_weight(self):
        return self.initial_weight

    def reset_initial_weight(self, new_weight):
        if not isinstance(new_weight, (int, float)):
            raise TypeError("new_weight must be an int or float")
        self.initial_weight = new_weight

    def add_attacker(self, attacker, weight=1.0):
        if not isinstance(attacker, Argument):
            raise TypeError("attacker must be an Argument instance")
        if not isinstance(weight, (int, float)):
            raise TypeError("attack weight must be a number")
        self.attackers[attacker] = weight

    def add_supporter(self, supporter, weight=1.0):
        if not isinstance(supporter, Argument):
            raise TypeError("supporter must be an Argument instance")
        if not isinstance(weight, (int, float)):
            raise TypeError("support weight must be a number")
        self.supporters[supporter] = weight

    def __repr__(self):
        attackers_names = ', '.join(a.name for a in self.attackers)
        supporters_names = ', '.join(s.name for s in self.supporters)
        return (
            f"Argument(name={self.name}, initial_weight={self.initial_weight}, "
            f"strength={self.strength}, attackers={{{attackers_names}}}, "
            f"supporters={{{supporters_names}}})"
        )

    def __str__(self):
        body = ', '.join(a.name for a in self.body)
        return f"{self.head} <- [{body}]"
    
    def __hash__(self):
        return hash((self.head, frozenset(self.body)))
    
    def __eq__(self, other):
        if not isinstance(other, Argument):
            return False
        return self.head == other.head and self.body == other.body and self.name == other.name
    