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
        self.arguments     = bag.get_arguments()
        self.assumptions   = self.arguments

        # now build the “set‐attacks” and “set‐supports”
        self.setAttacks    = defaultdict(list)
        self.setSupports   = defaultdict(list)
        for a in self.assumptions:
            self.setAttacks[a]  = []
            self.setSupports[a] = []
        for atk in bag.attacks:
            self.setAttacks[atk.attacked].append(atk.attacker.strength)
        for sup in bag.supports:
            self.setSupports[sup.supported].append(sup.supporter.strength)

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
        all_args = self.BAG.get_arguments()

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
        if verbose:
            if view == 'Arguments':
                print("iter\t"+"\t ".join(sorted([f"{arg.name}" for arg in self.assumptions])))
                print(str(count) + "\t" + "\t ".join([f"{round(state[arg], 3)}" for arg in sorted(self.assumptions, key=lambda arg: arg.name)]))
            elif view == 'Assumptions':
                current_strength = self.get_assumptions_strengths(assumptions, aggregate_strength_f)
                print("iter\t"+"\t ".join(sorted([f"{asm.name}" for asm in assumptions])))
                print(str(count) + "\t" + "\t ".join([f"{round(current_strength[1][asm], 3)}" for asm in sorted(assumptions, key=lambda asm: asm.name)]))
            else:
                raise ValueError("view must be either 'Arguments' or 'Assumptions'")

        while iterations > 0:
            count +=1
            if generate_plot:
                if view == 'Arguments':
                    for asm in self.assumptions:
                        self.graph_data[asm.name].append((count, state[asm]))
                elif view == 'Assumptions':
                    current_strength = self.get_assumptions_strengths(assumptions, aggregate_strength_f)
                    for asm in assumptions:
                        self.graph_data[asm.name].append((count, current_strength[1][asm]))
                else:
                    raise ValueError("view must be either 'Arguments' or 'Assumptions'")

            state = self.iterate(state)
            if verbose:
                if view == 'Arguments':
                    # print only 3 decimal places
                    print(str(count) + "\t" + "\t ".join([f"{round(state[arg], 3)}" for arg in sorted(self.assumptions, key=lambda arg: arg.name)]))
                elif view == 'Assumptions' and generate_plot:
                    # print only 3 decimal places
                    print(str(count) + "\t" + "\t ".join([f"{round(current_strength[1][asm], 3)}" for asm in sorted(assumptions, key=lambda asm: asm.name)]))
                else:
                    print("Not generating plot, so no need to print current strengths")
            iterations -= 1

            ## update strength values
            for asm in self.assumptions:
                asm.strength = state[asm]
            
        return state