import os
import re
import string
from BAG.Argument import Argument
from .Support import Support
from .Attack import Attack


class BAG:

    def __init__(self, path=None):
        """
        Bipolar Argumentation Graph:
        - arguments: dict mapping name -> Argument
        - attacks: list of Attack instances
        - supports: list of Support instances
        """
        self.arguments = {}
        self.attacks = []
        self.supports = []
        self.path = path

        if path:
            self._load_from_file(path)

    def _load_from_file(self, path):
        import os, re, string
        self.arguments.clear()
        self.attacks.clear()
        self.supports.clear()
        with open(os.path.abspath(path), 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                k_name = line.split('(')[0]
                args = re.findall(rf"{k_name}\((.*?)\)", line)[0].replace(' ', '').split(',')
                if k_name == 'arg':
                    argument = Argument(args[0], float(args[1]))
                    self.arguments[argument.name] = argument
                elif k_name == 'att':
                    attacker = self.arguments[args[0]]
                    attacked = self.arguments[args[1]]
                    self.add_attack(attacker, attacked)
                elif k_name == 'sup':
                    supporter = self.arguments[args[0]]
                    supported = self.arguments[args[1]]
                    self.add_support(supporter, supported)

    def add_attack(self, attacker, attacked, attack_weight=1):
        if not isinstance(attacker, Argument):
            raise TypeError("attacker must be of type Argument")
        if not isinstance(attacked, Argument):
            raise TypeError("attacked must be of type Argument")
        # register arguments
        self.arguments.setdefault(attacker.name, attacker)
        self.arguments.setdefault(attacked.name, attacked)
        # record
        attacked.add_attacker(attacker, attack_weight)
        self.attacks.append(Attack(attacker, attacked, attack_weight))

    def add_support(self, supporter, supported, support_weight=1):
        if not isinstance(supporter, Argument):
            raise TypeError("supporter must be of type Argument")
        if not isinstance(supported, Argument):
            raise TypeError("supported must be of type Argument")
        # register arguments
        self.arguments.setdefault(supporter.name, supporter)
        self.arguments.setdefault(supported.name, supported)
        # record
        supported.add_supporter(supporter, support_weight)
        self.supports.append(Support(supporter, supported, support_weight))

    def reset_strength_values(self):
        for arg in self.arguments.values():
            arg.strength = arg.initial_weight

    def get_arguments(self):
        return list(self.arguments.values())

    def __str__(self) -> str:
        # Arguments
        n_args = len(self.arguments)
        args_part = f"{n_args} arguments" if n_args > 10 else f"{list(self.arguments.keys())}"
        # Attacks
        n_atk = len(self.attacks)
        attacks_part = f"{n_atk} attacks" if n_atk > 10 else f"{self.attacks}"
        # Supports
        n_supp = len(self.supports)
        supports_part = f"{n_supp} supports" if n_supp > 10 else f"{self.supports}"

        return (
            f"BAG (path={self.path})\n"
            f"Arguments: {args_part}\n"
            f"Attacks: {attacks_part}\n"
            f"Supports: {supports_part}"
        )

    def __repr__(self) -> str:
        # Arguments repr or count
        n_args = len(self.arguments)
        args_repr = f"{n_args} args" if n_args > 10 else repr(self.arguments)
        # Attacks repr or count
        n_atk = len(self.attacks)
        attacks_repr = f"{n_atk} atks" if n_atk > 10 else repr(self.attacks)
        # Supports repr or count
        n_supp = len(self.supports)
        supports_repr = f"{n_supp} sups" if n_supp > 10 else repr(self.supports)

        return (
            f"BAG(path={self.path}) Arguments={args_repr} "
            f"Attacks={attacks_repr} Supports={supports_repr}"
        )
