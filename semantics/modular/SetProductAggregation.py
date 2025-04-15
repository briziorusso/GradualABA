class SetProductAggregation:
    def __init__(self):
        self.name = "SetProductAggregation"   

    def aggregate_set(self, set, state):
        result = 1
        for a in set:
            result *= 1-state[a]

        return result

    def __str__(self) -> str:
        return __class__.__name__
