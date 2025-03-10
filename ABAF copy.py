import sys
import os
from .Assumption import Assumptions
from .Sentence import Sentence
from .Rule import Rules

class ABAF:
    def __init__(self, path=None):
        self.assumptions = set()
        self.rules = []
        self.sentences = set()


class ABAF:
    def __init__(self, a = set(), c = dict(), rd = dict(),
            # q = list(), 
            s = set(), h = dict(), b = dict()):

        self.assumptions = a
        self.contrary = c
        self.rules_deriving = rd
        # self.queries = q
        self.sentences = s
        self.heads = h
        self.bodies = b

    # create an ABAF from a file, using the ICCMA format
    # INPUT RESTRICTIONS: a single contrary per assumption
    def create_from_file(self, framework_filename):
        with open(framework_filename, "r") as f:
            text = f.read().split("\n")
        for line in text:
            if line.startswith("a "):
                self.assumptions.add(str(line.split()[1]))
            if line.startswith("c "):
                components = line.split()
                self.contrary[str(components[1])] = [components[2]]

        # only one contrary!
        self.rules_deriving = {ctr_list[0] : [] for ctr_list in self.contrary.values() if ctr_list[0] not in self.assumptions}
        # Assumptions have empty set, used for SCC detection
        for asmpt in self.assumptions:
            self.rules_deriving[asmpt] = list()

        self.sentences.update(self.assumptions)
        # self.sentences.update(self.queries)
        print(self.sentences)
        for ctr_list in self.contrary.values():
            self.sentences.add(ctr_list[0])

        self.rule_indices = []
        self.heads = dict()
        self.bodies = dict()
        rule_index = 0
        for line in text:
            if line.startswith("r "):
                components = line.split()[1:]
                head, body = str(components[0]), components[1:]
                self.rule_indices.append(str(rule_index))
                self.heads[str(rule_index)] = head
                if head in self.rules_deriving:
                    self.rules_deriving[head].append(str(rule_index))
                else:
                    self.rules_deriving[head] = [str(rule_index)]

                self.bodies[str(rule_index)] = {str(b) for b in body}
                self.sentences.add(head)
                self.sentences.update(set(body))
                for b in body:
                    if not b in self.assumptions and not b in self.rules_deriving:
                        self.rules_deriving[b] = []

                rule_index += 1
    
    def print_ABA(self):
        # print ABAF
        print("Assumptions: ", self.assumptions)
        print("Contraries: ", self.contrary)
        print("Rules deriving: ", self.rules_deriving)
        print("Sentences: ", self.sentences)
        print("Heads: ", self.heads)
        print("Bodies: ", self.bodies)
