import networkx as nx
from ABAF import ABAF
import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import DEFAULT_WEIGHT


class DependencyGraph:
    def __init__(self):
        self.graph = nx.Graph() 

    def construct_from_abaf(self, abaf: ABAF):
        # Collect all sentences
        sentences = abaf.collect_sentences()

        # Add sentence nodes
        for sentence in sentences:
            node_type = 'sentence'
            for assumption in abaf.assumptions:
                if assumption.name == sentence.name:
                    node_type = 'assumption'
                elif assumption.contrary == sentence.name:
                    node_type = 'contrary'
            self.graph.add_node(sentence.name, type=node_type, weight=sentence.weight)

        # Add rule nodes and edges
        for asm in abaf.assumptions:
            if asm.contrary is not None:
                self.graph.add_edge(asm.name, asm.contrary, type='contrary')

        for rule in abaf.rules:
            rule_node = f"{rule.name}"
            self.graph.add_node(rule_node, type='rule', weight=rule.weight)
            for body_atom in rule.body:
                self.graph.add_edge(body_atom.name, rule_node, type='body')
            self.graph.add_edge(rule.head.name, rule_node, type='head')

    def get_graph(self):
        return self.graph

    def update_node_weight(self, node_name, new_weight):
        """Update the weight of a single node."""
        if node_name in self.graph.nodes:
            self.graph.nodes[node_name]['weight'] = new_weight
        else:
            raise ValueError(f"Node '{node_name}' does not exist in the graph.")

    def update_nodes_weights(self, weights_dict):
        """Update the weights of multiple nodes."""
        for node_name, new_weight in weights_dict.items():
            self.update_node_weight(node_name, new_weight)

    def update_abaf_weights(self, abaf: ABAF):
        """Update the weights of the given ABAF based on the graph."""
        for assumption in abaf.assumptions:
            if assumption.name in self.graph.nodes:
                assumption.weight = self.graph.nodes[assumption.name]['weight']
        for sentence in abaf.collect_sentences():
            if sentence.name in self.graph.nodes:
                sentence.weight = self.graph.nodes[sentence.name]['weight']
        for rule in abaf.rules:
            if rule.name in self.graph.nodes:
                rule.weight = self.graph.nodes[rule.name]['weight']

    def __str__(self):
        assumptions = [f"{node}[{data['weight']}]" for node, data in self.graph.nodes(data=True) if data.get('type') == 'assumption']
        contraries = [f"{node}[{data['weight']}]" for node, data in self.graph.nodes(data=True) if data.get('type') == 'contrary']
        rules = [f"{node}[{data['weight']}]" for node, data in self.graph.nodes(data=True) if data.get('type') == 'rule']
        sentences = [f"{node}[{data['weight']}]" for node, data in self.graph.nodes(data=True) if data.get('type') == 'sentence']
        
        contrary_edges = [(a, b) for a, b, data in self.graph.edges(data=True) if data.get('type') == 'contrary']
        head_edges = [(a, b) for a, b, data in self.graph.edges(data=True) if data.get('type') == 'head']
        body_edges = [(a, b) for a, b, data in self.graph.edges(data=True) if data.get('type') == 'body']

        result = "Assumption nodes:\n" + "\n".join(assumptions) + "\n\n"
        result += "Contrary nodes:\n" + "\n".join(contraries) + "\n\n"
        result += "Rule nodes:\n" + "\n".join(rules) + "\n\n"
        result += "Sentence nodes:\n" + "\n".join(sentences) + "\n\n"
        result += "Contrary edges:\n" + "\n".join([f"({a}, {b})" for a, b in contrary_edges]) + "\n\n"
        result += "Rule head edges:\n" + "\n".join([f"({a}, {b})" for a, b in head_edges]) + "\n\n"
        result += "Rule body edges:\n" + "\n".join([f"({a}, {b})" for a, b in body_edges]) + "\n"

        return result

    def draw(self):
        import matplotlib.pyplot as plt
        pos = nx.shell_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=10, font_weight='bold')
        edge_labels = nx.get_edge_attributes(self.graph, 'type')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        plt.show()