class SetMaxAggregation:
    def __init__(self):
        self.name = "SetMaxAggregation"   

    def aggregate_set(self, set, state):
        result = 0
        for a in set:
            if state[a] > result:
                result = state[a]
        return result

    def __str__(self) -> str:
        return __class__.__name__
