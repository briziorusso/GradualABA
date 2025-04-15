# Repos structure

gradualABA/
|-- algorithms/                         # Nicos approximations for continuous semantics
|   |-- Acyclic.py
|   |-- Approximator.py
|   |-- RK4.py
|-- semantics/
|   |-- modular                         # influence, aggregation, set-aggregation functions
|       |-- EulerBasedInfluence.py
|       |-- SetSumAggregation.py
|       |-- ProductAggregation.py
|       |-- ...
|   |-- ContinuousModularModel.py       # nico (for BAGs)
|   |-- QuadraticEnergyModel.py         # nico (for BAGs)
|   |-- Model.py                        # nico (for BAGs)
|   |-- bsafDiscreteModular.py          # for BSAFs 
|   |-- ! qbafDiscreteModular.py                                  # TODO!
|   |-- ! bsafContinuousModular.py                                # TODO!
|   |-- ! qbafContinuousModular.py                                # TODO!
|-- BAG/                                # QBAF class (nico)
|   |-- Argument.py
|   |-- Attack.py
|   |-- Support.py
|   |-- BAG.py
|-- BSAF/                               # BSAF class
|   |-- Argument.py
|   |-- BSAF.py
|-- ABAF/                               # ABAF class
|   |-- parser/
        |-- asp_parser.py
|   |-- Assumption.py
|   |-- Rule.py
|   |-- Sentence.py
|   |-- DependencyGraph.py
|   |-- ABAF.py
|-- constants.py
|-- test.py
|-- tryal.py
|-- requirements.txt
|-- README.md