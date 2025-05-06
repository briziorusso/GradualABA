from collections import defaultdict
from .bsafDiscreteModular import DiscreteModular  
from ABAF.Assumption import Assumption
from semantics.modular.ProductAggregation        import ProductAggregation
from semantics.modular.LinearInfluence           import LinearInfluence
from semantics.modular.SetProductAggregation     import SetProductAggregation
# import your BAG class
from BAG import BAG 

class DiscreteModularBAG(DiscreteModular):
    def __init__(self,
                 bag: BAG,
                 aggregation,
                 influence,
                 set_aggregation):
        
        self.BAG           = bag
        self.BSAF          = None

        self.graph_data    = defaultdict(list)
        self.aggregation   = aggregation
        self.influence     = influence
        self.set_aggregation = set_aggregation

        # treat every BAG.Argument as an “assumption”
        self.arguments = bag.get_arguments()
        self.assumptions = self.arguments
        
        # 0) create the reverse-lookup maps
        self._attackers  = defaultdict(list)
        self._supporters = defaultdict(list)

        # 1) build them from the BAG’s own Attacks/Supports
        for atk in bag.attacks:
            # atk.attacked is an Argument, atk.attacker too
            self._attackers[atk.attacked].append(atk.attacker)
        for sup in bag.supports:
            self._supporters[sup.supported].append(sup.supporter)

        # now build the “set‐attacks” and “set‐supports”
        self.setAttacks  = defaultdict(list)
        self.setSupports = defaultdict(list)
        for a in self.arguments:
            self.setAttacks[a]  = []
            self.setSupports[a] = []
        for atk in bag.attacks:
            # before: self.setAttacks[atk.attacked].append(atk.attacker.strength)
            # instead stash the attacker objects in a helper map:
            self._attackers.setdefault(atk.attacked, []).append(atk.attacker)
        for sup in bag.supports:
            self._supporters.setdefault(sup.supported, []).append(sup.supporter)
        
            # now initialize your numeric lists
        for a in self.arguments:
            self.setAttacks[a]  = [att.strength for att in self._attackers.get(a, [])]
            self.setSupports[a] = [sup.strength for sup in self._supporters.get(a, [])]

    def get_assumptions_strengths(self, assumptions, agg_strength_f = None) -> tuple:
        """
        Get raw & aggregated strengths for each assumption in `assumptions`.
        This version builds a claim→strengths map in one pass over the BAG arguments,
        so we don’t repeatedly scan the entire argument list for each assumption.
        """
        # 1) basic type checks
        if not isinstance(assumptions, set):
            raise TypeError("assumptions must be a set")
        if not all(isinstance(a, Assumption) for a in assumptions):
            raise TypeError("assumptions must be a set of Assumption instances")

        # 2) grab once
        all_args = self.arguments

        # 3) build claim → [strengths...]
        claim_map = defaultdict(list)
        for arg in all_args:
            claim_map[arg.claim].append(arg.strength)

        # 4) now for each requested assumption, look it up in O(1)
        strengths            = {}
        aggregated_strengths = {}
        for asm in assumptions:
            vals = claim_map.get(asm, [])
            if not vals:
                raise ValueError(f"No arguments claiming {asm!r} in the BAG")

            strengths[asm] = vals[:]  # copy if you plan to mutate

            if len(vals) > 1:
                # build the state once
                state = {i: s for i, s in enumerate(vals)}
                idxs  = set(state.keys())
                aggregated_strengths[asm] = agg_strength_f.aggregate_set(
                    set=idxs,
                    state=state
                )
            else:
                # just the single entry
                aggregated_strengths[asm] = vals[0]

        return strengths, aggregated_strengths

    def _print_row(self, it, state, agg_strengths, view, sorted_args, sorted_asms):
        # print header on first call
        if it == 0:
            if view == "Arguments":
                names = [a.name for a in sorted_args]
            else:
                names = [a.name for a in sorted_asms]
            print("iter\t" + "\t".join(names))

        # now print the values
        if view == "Arguments":
            vals = [round(state[a], 3) for a in sorted_args]
        else:
            if agg_strengths is None:
                # fallback to computing it on the fly
                _, agg_strengths = self.get_assumptions_strengths( set(sorted_asms), self.set_aggregation )
            vals = [round(agg_strengths[a], 3) for a in sorted_asms]

        row = [str(it)] + [f"{v:.3f}" for v in vals]
        print("\t".join(row))


    def solve(self, iterations, generate_plot=False, verbose=False, view='Arguments', assumptions=set(), aggregate_strength_f=None):

        if type(generate_plot) != bool:
            raise TypeError("generate_plot must be a boolean")

        if (type(iterations) != int):
            raise TypeError("iterations must be a float or integer")
        
        if self.BSAF is None and self.BAG is None:
            raise AttributeError("Model does not have BSAF or BAG attached")
        
        if view not in ['Arguments', 'Assumptions']:
            raise ValueError("view must be either 'Arguments' or 'Assumptions'")
        
        if view == 'Assumptions' and (not assumptions or not aggregate_strength_f):
            raise ValueError("Assumptions and aggregate_strength_f must be provided when view is 'Assumptions'")
        
        if verbose:
            print("\nDiscrete modular, iterations: ", iterations,"\n-------")
            print("Aggregation: ", self.aggregation.name)
            print("Influence: ", self.influence.name)
            print("Set Aggregation: ", self.set_aggregation.name)
            print("-------\n")

        state = {a: a.initial_weight for a in self.assumptions}
        count = 0
        sorted_args = sorted(self.arguments, key=lambda a: a.name)
        sorted_asms = sorted(assumptions or [], key=lambda a: a.name)
        if verbose:
            self._print_row(count, state, None, view, sorted_args, sorted_asms)

        while iterations > 0:
            count +=1
            if generate_plot:
                if view == 'Arguments':
                    for asm in self.assumptions:
                        self.graph_data[asm.name].append((count, state[asm]))
                elif view == 'Assumptions':
                    _, agg_strengths = self.get_assumptions_strengths(assumptions, aggregate_strength_f)
                    for asm in assumptions:
                        self.graph_data[asm.name].append((count, agg_strengths[asm]))
                else:
                    raise ValueError("view must be either 'Arguments' or 'Assumptions'")

            # 1) Update the state
            state = self.iterate(state)

            # 2) update the strengths of the arguments in the state
            for arg in self.arguments:
                arg.strength = state[arg]

            # now pull fresh strengths out of the stored attacker/supporter objects
            for a in self.arguments:
                self.setAttacks[a]  = [att.strength for att in self._attackers.get(a, [])]
                self.setSupports[a] = [sup.strength for sup in self._supporters.get(a, [])]
                
            _, agg_strengths = self.get_assumptions_strengths(assumptions, aggregate_strength_f)

            # 3) write them back onto the Assumption objects
            for asm in assumptions:
                asm.strength = agg_strengths[asm]

            if verbose:
                self._print_row(count, state, None, view, sorted_args, sorted_asms)

            iterations -= 1

            
        return state