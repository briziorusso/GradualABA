from .Rule import Rule
from .Assumption import Assumption
from .Sentence import Sentence
from BSAF.Argument import Argument
from BSAF.BSAF import BSAF
from BAG.BAG import BAG

from constants import DEFAULT_WEIGHT
from semantics.modular import SetProductAggregation as SPA, SetSumAggregation as SSA

import itertools
import clingo
import os, time
# ASP encoding for argument generation
ASP_ENCODING = """
    {in(X) : assumption(X)}.
    derived(X) :- assumption(X), in(X).
    derived(X) :- head(R,X), triggered_by_in(R).
    triggered_by_in(R) :- head(R,_), derived(X) : body(R,X).
    #show in/1.
    #show derived/1.
    """

class ABAF:
    def __init__(self, sentences=None, assumptions=None, rules=None, debug=False, arg_mode="basic", path=None):
        """
        Assumption-based Argumentation Framework:
        - assumptions: iterable of Assumption instances
        - rules: list of Rule instances
        """

        self.assumptions = set(assumptions) if assumptions else set()
        self.sentences = set(sentences) if sentences else set()
        self.rules = rules or []
        self.debug = debug
        self.arguments = []
        
        self.mode = arg_mode
        self.arg_counter = 1
        self.derived_in = dict()             # which arguments is head in, also asmpts (for basic -- otherwise all derivable)
        self.derives = dict()                # all derivable from args, also asmpts
        self.asmpt_derivable_from = dict()   # argument asmpt is derivable from, not including own support-asmpts
        self.used_in = dict()                # includes only support-asmpts, not derived assumptions
        self.support_of = dict()             # includes only support-asmpts, not derived assumptions
        self.current_atom = None
        self.last_model = None
        self.asmpt_to_singleton = dict()     # includes _the_ argument for singleton assumption set
        self.singleton_asp_atom = clingo.Function("singleton")

        if path:
            self._load_from_file(path)

        if not self.assumptions:
            raise ValueError("No assumptions provided. At least one assumption is required.")

    def _load_from_file(self, path):
        with open(path, "r") as f:
            text = f.read().split("\n")
        self.arguments.clear()
        self.sentences.clear()
        self.rules.clear()
        Rule.reset_identifiers()
        Sentence.reset_identifiers()
        Assumption.reset_identifiers()

        assumptions = set()
        sentences = set()
        contraries = dict()
        rules = []

        for line in text:
            if line.startswith("a "):
                assumptions.add(str(line.split()[1]))
            if line.startswith("c "):
                components = line.split()
                contraries[str(components[1])] = [components[2]]

        # only one contrary!
        rules_deriving = {ctr_list[0] : [] for ctr_list in contraries.values() if ctr_list[0] not in assumptions}
        # Assumptions have empty set, used for SCC detection
        for asmpt in assumptions:
            rules_deriving[asmpt] = list()
        #rules deriving now contains all the contrary values and all the assumptions

        sentences.update(assumptions) ## TODO: needed?
        for ctr_list in contraries.values():
            sentences.add(ctr_list[0])

        rule_indices = []
        heads = dict()
        bodies = dict()
        rule_index = 1
        for line in text:
            if line.startswith("r "):
                components = line.split()[1:]
                head, body = str(components[0]), components[1:]
                rule_indices.append(str(rule_index))
                heads[str(rule_index)] = head
                if head in rules_deriving:
                    rules_deriving[head].append(str(rule_index))
                else:
                    rules_deriving[head] = [str(rule_index)]

                bodies[str(rule_index)] = {str(b) for b in body}
                sentences.add(head)
                sentences.update(set(body))
                for b in body:
                    if not b in assumptions and not b in rules_deriving:
                        rules_deriving[b] = []

                rule_index += 1

        # create assumptions
        self.assumptions = set()
        for asmpt in assumptions:
            if asmpt in contraries:
                c = contraries[asmpt][0]
                assumption = Assumption(asmpt, contrary=c, initial_weight=DEFAULT_WEIGHT)
            else:
                assumption = Assumption(asmpt)
            self.assumptions.add(assumption)
            self.sentences.add(assumption)
        for sent in sentences:
            sent = Sentence(sent, initial_weight=DEFAULT_WEIGHT)
            self.sentences.add(sent)

        # create rules
        for rule_index in rule_indices:
            head = heads[rule_index]
            body = bodies[rule_index]
            ## check if head and body are in sentences and assumptions
            if head in [asmpt.name for asmpt in self.assumptions]:
                head_sent = next(a for a in self.assumptions if a.name == head)
            elif head in [s.name for s in self.sentences]:
                head_sent = next(s for s in self.sentences if s.name == head)
            else:
                head_sent = Sentence(head, initial_weight=DEFAULT_WEIGHT)
                self.sentences.add(head_sent)

            if body:
                body_sent = []
                for b in body:
                    if b in [asmpt.name for asmpt in self.assumptions]:
                        body_sent.append(next(a for a in self.assumptions if a.name == b))
                    elif b in [s.name for s in self.sentences]:
                        body_sent.append(next(s for s in self.sentences if s.name == b))
                    else:
                        body_sent.append(Sentence(b, initial_weight=DEFAULT_WEIGHT))
                        self.sentences.add(body_sent[-1])
            else:
                body_sent = []

            rule = Rule(head=head_sent, body=body_sent, name=f"r{rule_index}")
            rules.append(rule)

    def add_assumption(self, assumption: Assumption):
        if not isinstance(assumption, Assumption):
            raise TypeError("assumption must be an Assumption instance")
        self.assumptions.add(assumption)

    def add_rule(self, head: Sentence, body=None, name=None):
        body = body or []
        self.rules.append(Rule(head, body, name))

    def add_rules(self, rules):
        """
        Add multiple rules at once.
        """
        if not isinstance(rules, list):
            raise TypeError("rules must be a list of Rule instances")
        for rule in rules:
            if not isinstance(rule, Rule):
                raise TypeError("each rule must be a Rule instance")
            self.rules.append(rule)

    def collect_assumptions(self):
        """
        Collect all Assumption instances appearing in the bodies of rules.
        """
        for r in self.rules:
            for sent in r.body:
                if isinstance(sent, Assumption):
                    self.assumptions.add(sent)
        return set(self.assumptions)

    def collect_sentences(self):
        """
        Collect all Sentence instances used in heads and bodies of rules.
        """
        sentences = set()
        for r in self.rules:
            sentences.add(r.head)
            for sent in r.body:
                sentences.add(sent)
        return sentences

    def on_model(self, m):
        if True:
            print(m)
            print(f"{self.current_atom} <- {m}")
            print(f"{self.arg_counter}, {self.current_atom}")
        if True:
            self.derived_in[self.current_atom].add(self.arg_counter)
            self.derives[self.arg_counter] = {self.current_atom}
            self.support_of[self.arg_counter] = set()
            len_in = 0
            for asp_atom in m.symbols(shown=True):
                print("asp_atom", asp_atom)
                str_elem = str(asp_atom.arguments[0])
                if str_elem in [asm.name for asm in self.assumptions]:
                    if asp_atom.name == "in":
                        self.used_in[str_elem].add(self.arg_counter)
                        self.support_of[self.arg_counter].add(str_elem)
                        len_in += 1
                    elif asp_atom.name == "derived":
                        try:
                            self.asmpt_derivable_from[str_elem].add(self.arg_counter)
                        except KeyError:
                            self.asmpt_derivable_from[str_elem] = {self.arg_counter}
            print(self.support_of)

            if self.current_atom in [asm.name for asm in self.assumptions] and len_in == 1 and next(iter(self.support_of[self.arg_counter])) == self.current_atom:
                self.asmpt_to_singleton[self.current_atom] = self.arg_counter
            self.arg_counter += 1
            
   
    def on_model_all_derivations(self, m):
        self.last_model = m.symbols(shown=True)
        self.derives[self.arg_counter] = set()
        if True:
            self.support_of[self.arg_counter] = set()
            for asp_atom in m.symbols(shown=True):
                str_elem = str(asp_atom.arguments[0])
                if asp_atom.name == "in" and str_elem in self.assumptions:
                    self.used_in[str_elem].add(self.arg_counter)
                    self.support_of[self.arg_counter].add(str_elem)
                if asp_atom.name == "derived":
                    if str_elem not in self.derived_in:
                        self.derived_in[str_elem] = set()
                    self.derived_in[str_elem].add(self.arg_counter)
                    self.derives[self.arg_counter].add(str_elem)

            # Argument has only one assumption as assumption set
            if len(self.support_of[self.arg_counter]) == 1:
                self.asmpt_to_singleton[list(self.support_of[self.arg_counter])[0]] = self.arg_counter

            self.arg_counter += 1

    def generate_arguments_naive(self, atoms, asmpt_redundancy=False):
        for a in sorted(atoms):
            self.current_atom = a
            self.derived_in[a] = set()
            query_atom = clingo.Function("derived", [clingo.Function(a)])
            query_assumptions = [(query_atom,True)]

            if asmpt_redundancy:
                target_atom = clingo.Function("target", [clingo.Function(a)])
                self.ctl.assign_external(target_atom,True)

            self.ctl.solve(assumptions=query_assumptions, on_model=self.on_model)

            if asmpt_redundancy:
                self.ctl.assign_external(target_atom,False)

    def generate_arguments_supersets_out(self, atoms):
        for a in atoms:
            if a not in self.derived_in:
                self.derived_in[a] = set()

            query_atom = clingo.Function("derived", [clingo.Function(a)])
            query_assumption = [(query_atom,True)]

            while True:
                answer = self.ctl.solve(assumptions=query_assumption, on_model=self.on_model_all_derivations)
                if not answer.satisfiable:
                    break

                rule = []
                rule_str = []
                with self.ctl.backend() as backend:
                    for a in self.last_model:
                        if a.name == "in":
                            rule.append(backend.add_atom(a))
                            rule_str.append(a)
                        elif a.name == "not_derived":
                            rule.append(backend.add_atom(a))
                            rule_str.append(a)

                    backend.add_rule(head=[],body=rule)

    def _build_arguments(self, weight_agg):

        """
        Use Clingo to generate all arguments (singleton and derived), recording head and body.
        Returns:
          - arguments: list of Argument instances (with head, body attrs set)
          - arg_asms_map: dict mapping Argument -> list of Assumption instances supporting it
          - arg_head_map: dict mapping Argument -> claim string
        """
        print("Building arguments with ASP...")

        def parse_ASP(input_path="", ASP_input=None):
            atoms = set()
            assumptions = set()
            contraries = dict()
            # Read the ASP input file
            if os.path.exists(input_path):
                with open(input_path, "r") as infile:
                    ASP_input = infile.read().split("\n")
            for line in ASP_input:
                if line.startswith("assumption"):
                    assumptions.add(line.split("(")[1].split(")")[0])
                if line.startswith("head"):
                    atoms.add(line.split(",")[1].split(")")[0])
                if line.startswith("contrary"):
                    asm = line.split("(")[1].split(",")[0]
                    ctr = line.split(",")[1].split(")")[0]
                    if not asm in contraries:
                        contraries[asm] = list()
                    contraries[asm].append(ctr)

            # NOTE: atoms are those atoms that occur as head of a rule
            return atoms, assumptions, contraries

        # prepare ASP facts
        asp = []
        for asm in self.assumptions:
            asp.append(f"assumption({asm.name}).")
            if asm.contrary:
                asp.append(f"contrary({asm.name},{asm.contrary}).")
        for r in self.rules:
            rname = r.name or f"r_{r.head.name}"
            asp.append(f"head({rname},{r.head.name}).")
            for sent in r.body:
                asp.append(f"body({rname},{sent.name}).")
        asp.append(ASP_ENCODING)
        
        
        if self.mode == "basic":
            self.ctl = clingo.Control(arguments=[f"--models=0", "--heuristic=domain", "--enum-mode=domRec"])
        elif self.mode == "prune_supersets":
            self.ctl = clingo.Control()

        self.ctl.add("base", [], "\n".join(asp))

        atoms, assumptions, contraries = parse_ASP(ASP_input=asp)

        if self.mode == "basic":
            heuristic = ""
            for a in assumptions:
                heuristic += f"#heuristic in({a}). [1, false]\n"
            self.ctl.add("base", [], heuristic)

        self.ctl.ground([("base", [])])

        for asm in assumptions:
            self.used_in[asm] = set()

        # query = clingo.Function("derived", [clingo.Function("e")])
        asmpt_redundancy = False
        st = time.time()
        if self.mode == "basic":
            relevant_atoms = set(assumptions).union(set(atoms))
            # relevant_atoms.add(query)
            for ctrs in list(contraries.values()):
                relevant_atoms.update(ctrs)
            print(f"{len(relevant_atoms)} relevant atoms")
            self.generate_arguments_naive(relevant_atoms, asmpt_redundancy)
        elif self.mode == "prune_supersets":
            self.generate_arguments_supersets_out(atoms)

        for asm in assumptions:
            if asm in self.asmpt_to_singleton: continue

            if asm not in self.derived_in:
                self.derived_in[asm] = set()
            self.derived_in[asm].add(self.arg_counter)
            self.derives[self.arg_counter] = {asm}
            self.used_in[asm].add(self.arg_counter)
            self.support_of[self.arg_counter] = {asm}
            self.asmpt_to_singleton[asm] = self.arg_counter
            self.arg_counter += 1

        # create arguments
        arguments = []
        for i in range(1, self.arg_counter):
            head = self.derives[i]
            body_names = self.support_of[i]
            asm_objs = [next(a for a in self.assumptions if a.name == s) for s in body_names]
            support_weights = {asm.name: asm.initial_weight for asm in asm_objs}
            init_w = weight_agg.aggregate_set(weight_agg, state=support_weights, set=set(body_names))
            arg = Argument(name=f"arg{i}", initial_weight=init_w, head=next(iter(head)), body=asm_objs)
            arguments.append(arg)
            if self.debug:
                print(f"Argument {i}: {arg.name} ({arg.claim}) <- {arg.premise} ({len(arg.premise)})")

        print(f"{time.time()-st:.2f} seconds for argument construction - {self.arg_counter} arguments")

        self.arguments = arguments
        return arguments






    def build_arguments_procedure(self,weight_agg):
        """
        Generate all arguments (assumption arguments and derived arguments), with claim and premises
        Returns: 
            - list of tuples of arguments (premises, claim), where premises is a list of assumptions, claim is an assumption
        """
        print("Building arguments procedurally..")
        
        st = time.time()
        arguments = []


        for i in range(len(self.assumptions)):
            assumption_list = list(self.assumptions)
            arg = ([assumption_list[i]],assumption_list[i])
            arguments.append(arg)

        
        while True:
            new_args = 0

            for rule in self.rules:
                claim = rule.head

                all_arg_for_all_body_els = [[arg for arg in arguments if arg[1].name == sentence.name] for sentence in rule.body]
                if any(not candidate_for_body_el for candidate_for_body_el in all_arg_for_all_body_els):
                    pass
                all_top_subarguments = list(itertools.product(*all_arg_for_all_body_els))

                for top_subargs in all_top_subarguments:
                    tmp_supporting_assumptions = [x for xs in [item[0] for item in top_subargs] for x in xs]
                    supporting_assumptions = list(set(tmp_supporting_assumptions))
                    supporting_assumptions.sort(key=lambda x: x.name)

                    arg = (supporting_assumptions,claim)
                    if arg not in arguments:
                        arguments.append(arg)
                        new_args += 1
            
            if not new_args:
                break
        
        print("Arguments:")
        for arg in arguments:
            print([asm.name for asm in arg[0]], arg[1].name)


        argument_instances = []
        for arg in arguments:
            claim = arg[1]
            body_names = [asm.name for asm in arg[0]]
            asm_objs = [next(a for a in self.assumptions if a.name == s) for s in body_names]
            support_weights = {asm.name: asm.initial_weight for asm in asm_objs}
            init_w = weight_agg.aggregate_set(weight_agg, state=support_weights, set=set(body_names))
            weighted_arg = Argument(initial_weight=init_w, claim=claim, premise=asm_objs)
            argument_instances.append(weighted_arg)

        print("Arguments instances:")
        for arg in argument_instances:
            print(arg.name, arg.claim, [asm.name for asm in arg.premise], "initial_weight", arg.initial_weight)

        print(f"{time.time()-st:.2f} seconds for argument construction - {len(arguments)} arguments")

        self.arguments = argument_instances
        return argument_instances


    def _build_bag(self, weight_agg):
        """
        Build a BAG from this ABAF, using argument-level head/body:
          - support: if arg1.head matches any element in arg2.body ⇒ Support(arg1→arg2)
          - attack: if arg1.head equals contrary of any element in arg2.body ⇒ Attack(arg1→arg2)
        """
        args = self.build_arguments_procedure(weight_agg) if not self.arguments else self.arguments
        bag = BAG()
        # register arguments
        for arg in args:
            bag.arguments[arg.name] = arg
            bodylist = ",".join(premise.name for premise in arg.premise)
            print(f"{arg.name}: ([{bodylist}],{arg.claim})")
        print("Creting BAF: Extracting relations between arguments...")
        # build supports and attacks between arguments
        for a1 in args:
            for a2 in args:
                # skip self
                if a1 is a2:
                    continue
                # support: a1.head in names of a2.body
                if any(a1.head == premise.name for premise in a2.body):
                    bag.add_support(a1, a2)
                    if self.debug:
                        print(f"Support: {a1.name} -> {a2.name} ({a1.head} in {a2.body})")
                # attack: a2.body contains assumption with contrary == a1.head OR the contrary of a1.head is in a2.body
                if any(hasattr(premise, 'contrary') and premise.contrary == a1.head for premise in a2.body):
                    bag.add_attack(a1, a2)
                    if self.debug:
                        print(f"Attack: {a1.name} -> {a2.name} ({a1.head} = contrary of {a2.body})")
                elif any(hasattr(a1.head, 'contrary') and a1.head.contrary == premise for premise in a2.body):
                    bag.add_attack(a1, a2)
                    if self.debug:
                        print(f"Attack: {a1.name} -> {a2.name} ({a1.contrary} in {a2.body})")
        return bag

    def to_bag(self, weight_agg=SPA.SetProductAggregation):
        return self._build_bag(weight_agg)

    def to_bsaf(self, weight_agg=SPA.SetProductAggregation):
        args = self.build_arguments_procedure(weight_agg) if not self.arguments else self.arguments
        bsaf = BSAF(arguments=args, assumptions=self.assumptions)
        return bsaf

    def __repr__(self):
        asum = ",".join(sorted(a.name for a in self.assumptions))
        rules = ",".join(str(r) for r in self.rules)
        return f"ABAF(Assumptions=[{asum}], Rules=[{rules}])"

    def __str__(self):
        return self.__repr__()
