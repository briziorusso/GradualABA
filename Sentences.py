class Assumptions:
    def __init__(self, name, initial_weight, strength=None, attackers=None, supporters=None):
        self.name = name
        self.initial_weight = initial_weight
        self.strength = strength
        self.attackers = attackers
        self.supporters = supporters

        if type(initial_weight) != int and type(initial_weight) != float:
            raise TypeError("initial_weight must be of type integer or float")

        if strength is None:
            self.strength = initial_weight
