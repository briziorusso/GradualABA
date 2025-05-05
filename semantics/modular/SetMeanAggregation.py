class SetMeanAggregation:
    def __init__(self):
        self.name = "SetMeanAggregation"   

    def aggregate_set(self, set, state):
        """
        indices: iterable of keys into `state`
        state:    dict mapping each key -> numeric strength
        """
        # collect all the values
        vals = [state[i] for i in set]
        if not vals:
            # no elements → define as 0.0 (or whatever makes sense in your context)
            return 0.0
        if len(vals) == 1:
            # exactly one element → return it unchanged
            return vals[0]
        # multiple elements → return the mean
        return sum(vals) / len(vals)

    def __str__(self) -> str:
        return __class__.__name__
