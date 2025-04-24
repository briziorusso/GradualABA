from semantics.ContinuousModularModel import ContinuousModularModel
from BSAF.BSAF import BSAF
from BSAF.Argument import Argument
from ABAF.Assumption import Assumption
from ABAF.Rule import Rule
from ABAF.Sentence import Sentence
from ABAF.ABAF import ABAF

from constants import DEFAULT_WEIGHT

from semantics.modular.EulerBasedInfluence import EulerBasedInfluence
from semantics.modular.ProductAggregation import ProductAggregation
from semantics.modular.SetProductAggregation import SetProductAggregation
from semantics.modular.SetSumAggregation import SetSumAggregation

from semantics.bsafDiscreteModular import DiscreteModular


#### BAG
### Arguments
## ({a}, a)
## ({b}, b)
## ({c}, c)
## ({d}, d)
## ({d}, e)
## ({b,c}, f)
## ({d,c}, b)
## ({d,c}, f)

### Supports
## ({d,c}, b) -+ ({b}, b)

### Attacks
## ({b,c}, f) -> ({a}, a)
## ({d,c}, f) -> ({a}, a)

#### BSAF
### SetSupports
## {d,c} -+ b

### SetAttacks
## {b,c} -> a
## {d,c} -> a


## Assumptions
a = Assumption("a", initial_weight=0.8)
b = Assumption("b", initial_weight=0.5)
c = Assumption("c", initial_weight=0.5)
d = Assumption("d", initial_weight=0.5)
e = Sentence("e")
f = Sentence("f")
g = Sentence("g")

## Contraries
a.contrary = 'f'
# b.contrary = 'e'
# c.contrary = 'f'

## Rules
r1 = Rule(head=b, body=[e,c], name="r1")
# r2 = Rule(head=d, body=[b], name="r2")
r3 = Rule(head=e, body=[d], name="r3")
# att1 = Rule(head=d, body=[a], name="att1")
# att2 = Rule(head=a, body=[f], name="att2")
att3 = Rule(head=f, body=[b,c], name="att3")

abaf = ABAF(assumptions=[a,b,c,d], rules=[r1,r3,att3])

print(abaf)

bsaf = abaf.to_bsaf()

print(bsaf)

model = DiscreteModular(BSAF=bsaf, aggregation=ProductAggregation(), influence=EulerBasedInfluence(), set_aggregation=SetProductAggregation())

model.solve(10)

### AF implementation

bag = abaf.to_bag(weight_agg=SetProductAggregation)

print(bag) ### TODO: this has more supports than the example