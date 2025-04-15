import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import gradualABA as grad

# Define your model
model = grad.semantics.QuadraticEnergyModel()
# Set an approximator
model.approximator = grad.algorithms.RK4(model)

# set ABAF
# transform ABAF to BAG
# set BAG

model.BAG = grad.BAG("examples/stock_example.bag")

model.solve(delta=10e-2, epsilon=10e-6, verbose=True, generate_plot=False)

