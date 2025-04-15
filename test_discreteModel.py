from semantics.ContinuousModularModel import ContinuousModularModel
from BSAF.BSAF import BSAF
from BSAF.Argument import Argument  
from semantics.modular.EulerBasedInfluence import EulerBasedInfluence
from semantics.modular.ProductAggregation import ProductAggregation
from semantics.modular.SetProductAggregation import SetProductAggregation
from semantics.modular.SetSumAggregation import SetSumAggregation

from semantics.bsafDiscreteModular import DiscreteModular



a = Argument(initial_weight=0.2)
b = Argument(initial_weight=0.1)
c = Argument(initial_weight=1)
d = Argument()
e = Argument(initial_weight=0.5)

bsaf = BSAF(arguments=[a,b,c,d])

bsaf.add_attack([a],d)
bsaf.add_attack([c],a)
bsaf.add_attack([b],a)
print(bsaf)

model = DiscreteModular(BSAF=bsaf, aggregation=ProductAggregation(), influence=EulerBasedInfluence(), set_aggregation=SetSumAggregation())

model.solve(5)