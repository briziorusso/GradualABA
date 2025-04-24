from .Rule import Rule
from .Assumption import Assumption
from .Sentence import Sentence
from BSAF.Argument import Argument

from BSAF.BSAF import BSAF
from BAG.BAG import BAG

from constants import DEFAULT_WEIGHT
from semantics.modular import SetProductAggregation as SPA, SetSumAggregation as SSA

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
    def __init__(self, assumptions=None, rules=None, debug=False, arg_mode="basic"):
        """
        Abstract Bipolar Argumentation Framework:
        - assumptions: iterable of Assumption instances
        - rules: list of Rule instances
        """
        if assumptions is None:
            raise ValueError("assumptions parameter is required")
        self.assumptions = set(assumptions)
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
        if self.debug:
            print(m)
            print(f"{self.current_atom} <- {m}")
        if True:
            self.derived_in[self.current_atom].add(self.arg_counter)
            self.derives[self.arg_counter] = {self.current_atom}
            self.support_of[self.arg_counter] = set()
            len_in = 0
            for asp_atom in m.symbols(shown=True):
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
                print(f"Argument {i}: {arg.name} ({arg.head}) <- {arg.body} ({len(arg.body)})")

        print(f"{time.time()-st:.2f} seconds for argument construction - {self.arg_counter} arguments")

        self.arguments = arguments
        return arguments

    def _build_bag(self, weight_agg):
        """
        Build a BAG from this ABAF, using argument-level head/body:
          - support: if arg1.head matches any element in arg2.body ⇒ Support(arg1→arg2)
          - attack: if arg1.head equals contrary of any element in arg2.body ⇒ Attack(arg1→arg2)
        """
        args = self._build_arguments(weight_agg) if not self.arguments else self.arguments
        bag = BAG()
        # register arguments
        for arg in args:
            bag.arguments[arg.name] = arg
            bodylist = ",".join(premise.name for premise in arg.body)
            print(f"{arg.name}: ([{bodylist}],{arg.head})")
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
                elif any(hasattr(a1.head, 'contrary') and a1.contrary == premise for premise in a2.body):
                    bag.add_attack(a1, a2)
                    if self.debug:
                        print(f"Attack: {a1.name} -> {a2.name} ({a1.contrary} in {a2.body})")
        return bag

    def to_bag(self, weight_agg=SPA.SetProductAggregation):
        return self._build_bag(weight_agg)

    def to_bsaf(self, weight_agg=SPA.SetProductAggregation):
        args = self._build_arguments(weight_agg) if not self.arguments else self.arguments
        bsaf = BSAF(arguments=args, assumptions=self.assumptions)
        return bsaf

    def __repr__(self):
        asum = ",".join(sorted(a.name for a in self.assumptions))
        rules = ",".join(str(r) for r in self.rules)
        return f"ABAF(Assumptions=[{asum}], Rules=[{rules}])"

    def __str__(self):
        return self.__repr__()
