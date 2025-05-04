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

    def has_converged(self, epsilon: float, last_n: int) -> dict:
        """
        Look at each assumption’s last `last_n` strength values (from graph_data)
        and return a dict mapping assumption_name → bool indicating whether
        for every consecutive pair in that window the change ≤ epsilon.

        Requires that solve(..., generate_plot=True) has been called first.
        """
        if not isinstance(epsilon, (int, float)) or epsilon < 0:
            raise ValueError("epsilon must be a non-negative number")
        if not isinstance(last_n, int) or last_n < 2:
            raise ValueError("last_n must be an integer ≥ 2")
        if not self.graph_data:
            raise RuntimeError("No graph_data found — run solve(..., generate_plot=True) first")

        converged = {}
        for name, seq in self.graph_data.items():
            # need at least last_n points
            if len(seq) < last_n:
                converged[name] = False
                continue

            # grab just the last_n values
            last_vals = [val for (_, val) in seq[-last_n:]]
            # compute abs diffs between consecutive
            diffs = [abs(last_vals[i] - last_vals[i-1]) for i in range(1, last_n)]
            converged[name] = all(d <= epsilon for d in diffs)

        return converged

    def is_globally_converged(self, epsilon: float, last_n: int) -> bool:
        """
        True if *all* assumptions have converged over their last `last_n` steps.
        """
        conv_map = self.has_converged(epsilon, last_n)
        return all(conv_map.values())

    def convergence_time(self, epsilon: float, consecutive: int, out_mean: bool = False) -> dict:
        """
        For each assumption, scan its series in self.graph_data (a dict
        of name->[ (t, strength), … ]) and return the first time t at which
        the strength has changed by ≤ epsilon for `consecutive` successive steps.
        If an assumption never satisfies that, its value is None.

        :param epsilon: non-negative tolerance
        :param consecutive: how many consecutive small-change steps to require (>=1)
        :return: dict mapping assumption_name -> time_of_convergence (or None)
        """
        # 1) sanity checks
        if not isinstance(epsilon, (int, float)) or epsilon < 0:
            raise ValueError("epsilon must be a non-negative number")
        if not isinstance(consecutive, int) or consecutive < 1:
            raise ValueError("consecutive must be an integer ≥ 1")
        if not getattr(self, "graph_data", None):
            raise RuntimeError("No graph_data—run solve(..., generate_plot=True) first")

        times = {}
        for name, seq in self.graph_data.items():
            # need at least (consecutive+1) points to get `consecutive` diffs
            if len(seq) < consecutive + 1:
                times[name] = None
                continue

            # precompute absolute diffs between successive strength values
            diffs = [abs(seq[i][1] - seq[i-1][1]) for i in range(1, len(seq))]

            found = None
            # scan windows of length `consecutive`
            for start in range(0, len(diffs) - consecutive + 1):
                window = diffs[start : start + consecutive]
                if all(d <= epsilon for d in window):
                    # we take the time‐stamp at the *end* of that window:
                    # that's seq[start + consecutive][0]
                    found = seq[start + consecutive][0]
                    break

            times[name] = found
        
        # if out_mean is True, return the mean of the times
        if out_mean:
            mean_times = []
            for name, time in times.items():
                if time is not None:
                    mean_times.append(time)
            if len(mean_times) > 0:
                return sum(mean_times) / len(mean_times)
            else:
                return None

        return times


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
        
        if verbose:
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
    
    def __repr__(self) -> str:
        return f"DiscreteModular(BSAF={self.BSAF}, aggregation={self.aggregation}, influence={self.influence}, set_aggregation={self.set_aggregation})"
    
    def __str__(self) -> str:
        return f"DiscreteModular - BSAF: {self.BSAF}, Aggregation: {self.aggregation}, Influence: {self.influence}, Set Aggregation: {self.set_aggregation})"