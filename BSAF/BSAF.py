from .Argument import Argument
# from ABAF.Assumption import Assumption

class BSAF:
    def __init__(self, arguments, assumptions):
        """
        Bipolar Set Argumentation Framework mapping sets of assumptions to single assumption targets.

        Args:
          - assumptions: iterable of Assumption instances
          - arg_asms_map: dict mapping Argument -> list of Assumption instances supporting it
          - arg_head_map: dict mapping Argument -> claim string

        After initialization, self.supports and self.attacks map each assumption to a set of frozensets
        (each frozenset is a coalition of assumptions supporting or attacking the key assumption).
        """
        # Validate inputs
        if assumptions is None or arguments is None:
            raise ValueError("assumptions and arguments are required")
        # Store unique assumptions
        self.assumptions = set(assumptions)
        self.arguments = set(arguments)
        # Initialize empty relations
        self.supports = {asm: set() for asm in self.assumptions}
        self.attacks  = {asm: set() for asm in self.assumptions}
        print("Creating BSAF: Extracting relations from arguments...")
        # Extract relations from arguments
        for arg in self.arguments:
            coalition = frozenset(arg.premise)
            ## Filter assumption arguments ({a},a) and sets that contain a sentence that is not assumption (i.e. it can be derived)
            if (len(coalition) == 1 and arg.claim in [a.name for a in arg.premise]): # or not all(a for a in coalition if a in self.assumptions):
                continue
            claim = arg.claim
            # SUPPORT: argument's claim matches assumption name
            target = next((a for a in self.assumptions if a == claim), None)
            if target:
                self.supports[target].add(coalition)
                continue
            # ATTACK: argument's claim matches assumption.contrary
            attacked = next((a for a in self.assumptions if a.contrary == claim.name), None)
            if attacked and claim is not None:
                self.attacks[attacked].add(coalition)

    def add_attack(self, attackers, attacked):
        """
        attackers: iterable of Argument instances
        attacked: single Argument instance
        Records that attackers attack the attacked argument.
        Only attackers that are in the assumptions. 
        """
        if attacked not in self.arguments:
            raise ValueError("attacked argument not in framework")
        # Filter valid assumption-based attackers
        valid = frozenset(asm for asm in self.assumptions if asm.name in [a.name for a in attackers])
        if not valid:
            return
        attacked_asm = next((asm for asm in self.assumptions if asm.name == attacked.name), None)

        self.attacks[attacked_asm].add(valid)

    def add_support(self, supporters, supported):
        """
        supporters: iterable of Argument instances
        supported: single Argument instance
        Records that supporters support the supported argument.
        Only supporters that are in the assumptions.
        """
        if supported not in self.arguments:
            raise ValueError("supported argument not in framework")
        valid = frozenset(asm for asm in self.assumptions if asm.name in [a.name for a in supporters])
        if not valid:
            return
        supported_asm = next((asm for asm in self.assumptions if asm.name == supported.name), None)
        self.supports[supported_asm].add(valid)

    def __repr__(self):
        asum = sorted(a.name for a in self.assumptions)
        sup_parts = [f"{asm.name}:[{','.join(sorted(c.name for c in coal))}]" \
                     for asm, cols in self.supports.items() for coal in cols]
        atk_parts = [f"{asm.name}:[{','.join(sorted(c.name for c in coal))}]" \
                     for asm, cols in self.attacks.items() for coal in cols]
        return (
            f"BSAF(Assumptions={asum}, "
            f"\nSupports={{" + ",".join(sup_parts) + "}}, "
            f"\nAttacks={{" + ",".join(atk_parts) + "}})"
        )

    def __str__(self):
        lines = []
        for asm in sorted(self.assumptions, key=lambda x: x.name):
            for coal in self.supports.get(asm, []):
                names = ",".join(sorted(a.name for a in coal))
                lines.append(f"\nSupport: {{{names}}} -> {asm.name}")
            for coal in self.attacks.get(asm, []):
                names = ",".join(sorted(a.name for a in coal))
                lines.append(f"\nAttack:  {{{names}}} -> {asm.name}")
        return " ".join(lines)