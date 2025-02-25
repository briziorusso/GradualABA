import networkx as nx

class DependencyGraph:
    def __init__(self, a = set(), s = set(), r = set(), c = set(), h = set(), b = set()):

        self.assumption_nodes = a
        self.sentence_nodes = s
        self.rule_nodes = r
        self.contrary_arcs = c
        self.head_arcs = h
        self.body_arcs = b

        # self.graph = nx.DiGraph()  # Directed graph to represent dependencies