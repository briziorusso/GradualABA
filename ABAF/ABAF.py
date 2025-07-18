from .Rule import Rule
from .Assumption import Assumption
from .Sentence import Sentence
from BSAF.Argument import Argument
from BSAF.BSAF import BSAF
from BAG.BAG import BAG

from constants import DEFAULT_WEIGHT
from semantics.modular.SetProductAggregation import SetProductAggregation 

from collections import defaultdict
from tqdm import tqdm
import itertools
import clingo
import os, time
import random
import hashlib
# ASP encoding for argument generation
ASP_ENCODING = """
    {in(X) : assumption(X)}.
    derived(X) :- assumption(X), in(X).
    derived(X) :- head(R,X), triggered_by_in(R).
    triggered_by_in(R) :- head(R,_), derived(X) : body(R,X).
    #show in/1.
    #show derived/1.
    """

def my_weight():
    return random.random() if random.random() < 0.1 else 0.2

def stable_hash(s:str) -> int:
    h = hashlib.sha256(s.encode("utf8")).digest()
    # take e.g. the first two bytes as an int 0..65535
    return (h[0]<<8) | h[1]

class ABAF:
    def __init__(self, sentences=None, assumptions=None, rules=None, debug=False, arg_mode="basic", path=None,
        default_weight: float = DEFAULT_WEIGHT, weight_fn = None, seed=42):
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

        self.default_weight = default_weight
        self.weight_fn      = weight_fn if weight_fn else None
        self.seed = seed

        if path:
            self._load_from_file(path)

        if not self.assumptions:
            raise ValueError("No assumptions provided. At least one assumption is required.")

        # --- Flatness check: no assumption may appear as the head of a rule ---
        assump_names = {a.name for a in self.assumptions}
        self.non_flat = any([rule.head.name in assump_names for rule in self.rules])

    def _load_from_file(self, path):

        random.seed(self.seed)

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
                if contraries.get(str(components[1])) is not None:
                    contraries[str(components[1])].append(components[2])
                else:
                    contraries[str(components[1])] = [components[2]]
        ## check that each assumption has at most one contrary
        for asmpt in assumptions:
            if asmpt in contraries:
                if len(set(contraries[asmpt])) > 1:
                    raise ValueError(f"Assumption {asmpt} has more than one contrary: {set(contraries[asmpt])}")

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

        # create assumptions with the new weighting scheme:
        self.assumptions = set()
        for asmpt in sorted(assumptions):
            ## make sure that the weight changes for each assumption
            seed = self.seed + stable_hash(asmpt) % 1000
            random.seed(seed)
            w = self.weight_fn() if self.weight_fn is not None else self.default_weight
            if asmpt in contraries:
                c = contraries[asmpt][0]
                assumption = Assumption(asmpt, contrary=c, initial_weight=w)
            else:
                assumption = Assumption(asmpt, initial_weight=w)
            self.assumptions.add(assumption)

        # create sentences with the same scheme:
        for sent_name in sorted(sentences):
            ## make sure that the weight changes for each sentence
            seed = self.seed + stable_hash(sent_name) % 1000
            random.seed(seed)
            w = self.weight_fn() if self.weight_fn is not None else self.default_weight
            sent = Sentence(sent_name, initial_weight=w)
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
                # make sure that the weight changes for each sentence
                seed = self.seed + stable_hash(head) % 1000
                random.seed(seed)
                w = self.weight_fn() if self.weight_fn is not None else self.default_weight
                head_sent = Sentence(head, initial_weight=w)
                self.sentences.add(head_sent)

            if body:
                body_sent = []
                for b in body:
                    if b in [asmpt.name for asmpt in self.assumptions]:
                        body_sent.append(next(a for a in self.assumptions if a.name == b))
                    elif b in [s.name for s in self.sentences]:
                        body_sent.append(next(s for s in self.sentences if s.name == b))
                    else:
                        # make sure that the weight changes for each sentence
                        seed = self.seed + stable_hash(b) % 1000
                        random.seed(seed)
                        w = self.weight_fn() if self.weight_fn is not None else self.default_weight
                        new_sent = Sentence(b, initial_weight=w)
                        body_sent.append(new_sent)
                        self.sentences.add(new_sent)
            else:
                body_sent = []

            rule = Rule(head=head_sent, body=body_sent, name=f"r{rule_index}")
            self.rules.append(rule)

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
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

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
            init_w = weight_agg.aggregate_set(state=support_weights, set=set(body_names))
            arg = Argument(name=f"arg{i}", initial_weight=init_w, claim=next(iter(head)), premise=asm_objs)
            arguments.append(arg)
            if self.debug:
                print(f"Argument {i}: {arg.name} ({arg.claim}) <- {arg.premise} ({len(arg.premise)})")

        print(f"{time.time()-st:.2f} seconds for argument construction - {self.arg_counter} arguments")

        self.arguments = arguments
        return arguments

    def build_arguments_procedure(self, weight_agg):
        """
        Generate all arguments (assumption arguments and derived arguments),
        returning a list of Argument instances.
        """
        # print("Building arguments procedurally…")
        start = time.time()

        # 1) Build name→Assumption and name→Sentence mappings
        name2asm = {asm.name: asm for asm in self.assumptions}

        # map *all* sentences (assumptions + any rule heads/bodies) to their objects
        name2sent = dict(name2asm)  
        for rule in self.rules:
            # head
            name2sent[rule.head.name] = rule.head
            # body literals
            for lit in rule.body:
                name2sent[lit.name] = lit

        # 2) Seed with single-assumption arguments
        seen       = set()   # Set[ (tuple(sorted_premise_names), claim_name) ]
        key_list   = []      # preserve insertion order

        for asm in self.assumptions:
            k = ((asm.name,), asm.name)
            seen.add(k)
            key_list.append(k)

        # 3) Expand until fixpoint
        while True:
            new_keys = []

            # index existing by claim for fast lookup
            by_claim = defaultdict(list)
            for prem_names, claim_name in key_list:
                by_claim[claim_name].append(prem_names)

            for rule in tqdm(self.rules, desc="Analysing rules"):
                # skip if any body atom has no arguments yet
                if any(lit.name not in by_claim for lit in rule.body):
                    continue

                # get lists of premise‐tuple choices for each body atom
                pools = [by_claim[lit.name] for lit in rule.body]

                # combine one sub‐arg per body atom
                for combo in itertools.product(*pools):
                    merged = tuple(sorted(set().union(*combo)))
                    k = (merged, rule.head.name)
                    if k not in seen:
                        seen.add(k)
                        new_keys.append(k)

            if not new_keys:
                break

            key_list.extend(new_keys)

        # 4) Now instantiate *once* per unique key
        argument_instances = []
        for prem_names, claim_name in key_list:
            premise_objs = [name2sent[n] for n in prem_names]
            # only assumptions carry initial_weight
            state = {n: name2asm[n].initial_weight for n in prem_names}

            init_w = weight_agg.aggregate_set(
                state=state,
                set=set(prem_names)
            )

            arg = Argument(
                claim=name2sent[claim_name],
                premise=premise_objs,
                initial_weight=init_w
            )
            argument_instances.append(arg)

        elapsed = time.time() - start
        print(f"{elapsed:.2f}s to build {len(argument_instances)} arguments")

        if self.debug:
            for arg in argument_instances:
                names = [p.name for p in arg.premise]
                print(f"{arg.name}: {arg.claim.name} ← {names} (w={arg.initial_weight})")

        self.arguments = argument_instances
        return argument_instances

    def build_arguments_procedure_og(self,weight_agg):
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

            for rule in tqdm(self.rules, desc="Analysing rules"):
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
        
        if self.debug:
            print("Arguments:")
            for arg in arguments:
                print([asm.name for asm in arg[0]], arg[1].name)


        argument_instances = []
        for arg in arguments:
            claim = arg[1] ## THIS NEEDS TO BE A SENTENCE
            body_names = [asm.name for asm in arg[0]]
            asm_objs = [next(a for a in self.assumptions if a.name == s) for s in body_names]
            support_weights = {asm.name: asm.initial_weight for asm in asm_objs}
            init_w = weight_agg.aggregate_set(state=support_weights, set=set(body_names))
            weighted_arg = Argument(initial_weight=init_w, claim=claim, premise=asm_objs)
            argument_instances.append(weighted_arg)

        if self.debug:
            print("Arguments instances:")
            for arg in argument_instances:
                print(arg.name, arg.claim, [asm.name for asm in arg.premise], "initial_weight", arg.initial_weight)

        print(f"{time.time()-st:.2f} seconds for argument construction - {len(arguments)} arguments")

        self.arguments = argument_instances
        return argument_instances


    def _build_bag(self, weight_agg, args=None):
        """
        Build a BAG from this ABAF, using argument-level head/body:
          - support: if arg1.head matches any element in arg2.body ⇒ Support(arg1→arg2)
          - attack: if arg1.head equals contrary of any element in arg2.body ⇒ Attack(arg1→arg2)
        """
        if args is None:
            args = self.build_arguments_procedure(weight_agg) if not self.arguments else self.arguments
        bag = BAG()

        # alias methods and functions locally for speed
        addsup = bag.add_support
        addatk = bag.add_attack
        agg_set = weight_agg.aggregate_set

        # register arguments to ``
        for arg in args:
            ## update initial weight and strength to be the weight_agg of the premises initial weights
            arg.initial_weight = agg_set(state={asm.name: asm.initial_weight for asm in arg.premise}, 
                                                          set=set(a.name for a in arg.premise))
            arg.strength = arg.initial_weight
            bag.arguments[arg.name] = arg
            
            if self.debug:
                premiselist = ",".join(premise.name for premise in arg.premise)
                print(f"{arg.name}: ([{premiselist}],{arg.claim})")

        if self.debug:
            print("Creating BAF: Extracting relations between arguments...")
        premise_map = defaultdict(list)
        contrary_map = defaultdict(set)
        for arg in args:
            for asm in arg.premise:
                premise_map[asm.name].append(arg)
                if hasattr(asm, "contrary"):
                    contrary_map[asm.contrary].add(asm)
        

        # 3) now only iterate the truly necessary pairs
        for a1 in args:
            claim_name = a1.claim.name

            # a1 supports every a2 whose premise contains claim_name
            for a2 in premise_map.get(claim_name, ()):
                if a2 is not a1:
                    addsup(a1, a2)

            # a1 attacks every a2 whose premise is the contrary of a1.claim, if any
            c = contrary_map.get(claim_name, None) 
            if c:
                if len(c) > 1:
                    raise ValueError(f"Multiple contraries for {claim_name}: {c}")
                c = next(iter(c))
                for a2 in premise_map.get(c.name, ()):
                    if a2 is not a1:
                        addatk(a1, a2)
        return bag

    def to_bag(self, weight_agg=SetProductAggregation(), args=None):
        return self._build_bag(weight_agg, args)

    def to_bsaf(self, weight_agg=SetProductAggregation(), args=None):
        if args is None:
            args = self.build_arguments_procedure(weight_agg) if not self.arguments else self.arguments
        bsaf = BSAF(arguments=args, assumptions=self.assumptions)
        return bsaf

    def __repr__(self):
        asum = ",".join(sorted(a.name for a in self.assumptions))
        rules = ",".join(str(r) for r in self.rules)
        return f"ABAF(Assumptions=[{asum}], Rules=[{rules}])"

    def __str__(self):
        return self.__repr__()
