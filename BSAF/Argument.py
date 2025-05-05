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
      - Optional claim (claim) and premise (supporting assumptions) attributes
    """
    _counter = 0
    _instances = {}

    def __new__(cls, name=None, initial_weight=DEFAULT_WEIGHT, strength=None, claim=None, premise=[]):
        # Generate unique name if not provided
        if name is None:
            name = f"x{cls._counter + 1}"
            cls._counter += 1
        # Prevent duplicate names
        if name in cls._instances:
            # If same content, return existing
            if cls._instances[name].claim is claim and cls._instances[name].premise == premise:
                return cls._instances[name]
            # otherwise bump the counter and fall through to create new
            name = f"x{cls._counter + 1}"
            cls._counter += 1

        inst = super().__new__(cls)
        cls._instances[name] = inst
        inst.name = name
        return inst

    def __init__(self, name=None, initial_weight=DEFAULT_WEIGHT, strength=None, 
                 premise=None, claim=None, attackers=None, supporters=None):
        if hasattr(self, '_initialized'):
            return
        if not isinstance(initial_weight, (int, float)):
            raise TypeError("initial_weight must be an int or float")
        
        if name is not None:
            self.name = name
        self.initial_weight = initial_weight
        self.strength = strength
        self.attackers = attackers
        self.supporters = supporters

        if strength is None:
            self.strength = initial_weight

        if attackers is None:
            self.attackers = {}

        if supporters is None:
            self.supporters = {}

        self._initialized = True
        self.claim = claim
        self.premise = list(premise) if premise is not None else []

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

    def get_name(self):
        return self.name
    
    def get_initial_weight(self):
        return self.initial_weight
    
    def __repr__(self):
        premise = ', '.join(a.name for a in self.premise)
        if len(self.premise) == 0 and self.claim is None:
            return f"Argument {self.name}: initial weight {self.initial_weight}, strength {self.strength}, attackers {self.attackers}, supporters {self.supporters}"
        return f"({[premise]},{self.claim})"

    def __str__(self):
        premise = ', '.join(a.name for a in self.premise)
        if len(self.premise) == 0 and self.claim is None:
            return f"Argument(name={self.name}, weight={self.initial_weight}, strength={self.strength})"
        return f"({[premise]},{self.claim})"

    def __hash__(self):
        return hash((self.claim, frozenset(self.premise)))

    def __eq__(self, other):
        if not isinstance(other, Argument):
            return False
        return self.claim == other.claim and set(self.premise) == set(other.premise) and self.name == other.name

    def __reduce__(self):
        ## To pickle the object, we need to provide a callable and its arguments
        return (
            self.__class__,
            (self.name,
             self.initial_weight,
             self.strength,
             list(self.premise),
             self.claim)
        )

    def __reset__(self):
        self.strength = self.initial_weight
        self.attackers.clear()
        self.supporters.clear()
        self.premise.clear()
        self.claim = None

    @classmethod
    def reset_identifiers(cls):
        """Reset the used identifiers (for testing or reloading purposes)."""
        cls._instances.clear()