class ProductAggregation:
    def __init__(self):
        self.name = "ProductAggregation"   

    def aggregate_strength(self, attacker_vals, supporter_vals):
        support_value = 1
        for a in attacker_vals:
            support_value *= 1-a

        attack_value = 1
        for s in supporter_vals:   
            attack_value *= 1-s

        return support_value - attack_value

    def __str__(self) -> str:
        return __class__.__name__
