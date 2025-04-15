from BSAF.Argument import Argument
from BSAF.BSAF import BSAF
from BSAF.BSAF import sort_by_key

# Create arguments with explicit names and weights
arg1 = Argument("A1", initial_weight=0.5)
arg2 = Argument("A2", initial_weight=0.7)

# Create arguments without names (names will be auto-generated)
arg3 = Argument(initial_weight=0.9)
arg4 = Argument(initial_weight=0.3)

arg5 = Argument("arg")

# Create an argument with a duplicate name (reuses the existing instance)
# arg5 = Argument("A1", initial_weight=1.0)

# Print arguments
print(arg1)  # Output: Argument(name=A1, weight=0.5, strength=0.5)
print(arg2)  # Output: Argument(name=A2, weight=0.7, strength=0.7)
print(arg3)  # Output: Argument(name=x1, weight=0.9, strength=0.9)
print(arg4)  # Output: Argument(name=x2, weight=0.3, strength=0.3)
# print(arg5)  # Output: Argument(name=A1, weight=0.5, strength=0.5)

bsaf = BSAF(arguments=[arg1, arg2, arg3, arg3])

print(bsaf)

bsaf.add_argument(arg4)
bsaf.add_argument(arg1)

print([arg.name for arg in bsaf.arguments])


# add attack
bsaf.add_attack([arg3, arg2], arg2)
bsaf.add_attack([arg1, arg3, arg2], arg1)
bsaf.add_attack([arg1], arg3)
bsaf.add_attack([arg1], arg3)

bsaf.add_support([arg1], arg2)
bsaf.add_support([arg1], arg5)

print("here we go\n!", bsaf)

