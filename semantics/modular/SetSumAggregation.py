class SetSumAggregation:
    def __init__(self):
        self.name = "SetSumAggregation"   

    def aggregate_set(self, set, state):
        result = 0
        for a in set:
            result += state[a]

        return result

    def __str__(self) -> str:
        return __class__.__name__
