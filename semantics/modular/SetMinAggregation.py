class SetMinAggregation:
    def __init__(self):
        self.name = "SetMinAggregation"   

    def aggregate_set(self, set, state):
        result = 1
        for a in set:
            if state[a] < result:
                result = state[a]
        return result

    def __str__(self) -> str:
        return __class__.__name__
