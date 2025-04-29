class SumAggregation:
    def __init__(self) -> None:
        self.name = "SumAggregation"
        pass

    def aggregate_strength(self, attackers, supporters):
        aggregate = 0
        for a in attackers:  
                aggregate -= a

        for s in supporters:
                aggregate += s

        return aggregate
    
    def __str__(self) -> str:
        return __class__.__name__
