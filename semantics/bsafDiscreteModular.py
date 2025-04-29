from collections import defaultdict

class DiscreteModular:
    def __init__(self, BSAF=None, aggregation=None, influence=None, set_aggregation=None):
        self.graph_data = defaultdict(list)
        self.BSAF = BSAF
        self.aggregation = aggregation
        self.influence = influence
        self.set_aggregation = set_aggregation

        # take values from BSAF
        self.arguments = BSAF.arguments
        self.assumptions = BSAF.assumptions
        self.setAttacks = BSAF.attacks
        self.setSupports = BSAF.supports

        # self.arguments = [a,b,c]
        # self.setAttacks = {a: [[1,0,1],[0,1,1]], b: [0,1,1]], c: [[1,1,0]]}

    def iterate(self, state):

        # computes the next state
        next_state = {}

        # aggregate the set-attacks and set-supports using set_aggregation
        aggregated_setAttacks = {asm: [] for asm in self.assumptions}
        aggregated_setSupports = {asm: [] for asm in self.assumptions}    

        for assumption in self.assumptions:
            att_aggregation = []
            for attack in self.setAttacks[assumption]:
                set_aggregation = self.set_aggregation.aggregate_set(attack, state)
                att_aggregation.append(set_aggregation)
            aggregated_setAttacks[assumption] = att_aggregation

            sup_aggregation = []
            for support in self.setSupports[assumption]:
                set_aggregation = self.set_aggregation.aggregate_set(support, state)
                sup_aggregation.append(set_aggregation)
            aggregated_setSupports[assumption] = sup_aggregation

        # compute the next state
        for a in self.assumptions:
            aggregate_strength = self.aggregation.aggregate_strength(aggregated_setAttacks[a], aggregated_setSupports[a])
            result = self.influence.compute_strength(a.initial_weight, aggregate_strength)

            next_state[a] = result


        return next_state


    def solve(self, iterations, generate_plot=False, verbose=False):

        if type(generate_plot) != bool:
            raise TypeError("generate_plot must be a boolean")

        if (type(iterations) != int):
            raise TypeError("iterations must be a float or integer")
        
        if self.BSAF is None:
            raise AttributeError("Model does not have BAG attached")
        
        print("\nDiscrete modular, iterations: ", iterations,"\n-------")
        print("Aggregation: ", self.aggregation.name)
        print("Influence: ", self.influence.name)
        print("Set Aggregation: ", self.set_aggregation.name)
        print("-------\n")

        state = {a: a.initial_weight for a in self.assumptions}
        count = 0
        if verbose:
            print("iter\t"+"\t ".join(sorted([f"{arg.name}" for arg in self.assumptions])))
            print(str(count) + "\t" + "\t ".join([f"{round(state[arg], 3)}" for arg in sorted(self.assumptions, key=lambda arg: arg.name)]))

        while iterations > 0:
            count +=1
            if generate_plot:
                for asm in self.assumptions:
                    self.graph_data[asm.name].append((count, state[asm]))
            state = self.iterate(state)
            if verbose:
                # print only 3 decimal places
                print(str(count) + "\t" + "\t ".join([f"{round(state[arg], 3)}" for arg in sorted(self.assumptions, key=lambda arg: arg.name)]))
            iterations -= 1

        return state

    def __repr__(self, name) -> str:
        return f"{name}({self.BAG}, {self.approximator}, {self.arguments}, {self.argument_strength}, {self.attacker}, {self.supporter})"

    def __str__(self, name) -> str:
        return f"{name} - BAG: {self.BAG}, Approximator: {self.approximator}, Arguments: {self.arguments}, Argument strength: {self.argument_strength}, Attacker: {self.attacker}, Supporter: {self.supporter})"
