import random
import argparse
import os, sys

BASE_SEED = 42

def create_depth_bounded_framework(n_sentences, n_assumptions, n_rules_per_head,
    size_of_bodies, nonflat_coef, d_bound):

    assumptions = ["a" + str(i) for i in range(n_assumptions)]
    sentences = ["s" + str(i) for i in range(n_sentences-n_assumptions)]

    contraries = {asmpt: random.choice(sentences+assumptions) for asmpt in assumptions}

    rules = []

    head_pool = None
    if nonflat_coef > 0:
        head_pool = sentences+random.sample(assumptions, int(len(assumptions)*nonflat_coef))
        #head_pool = sentences+assumptions
    else:
        head_pool = sentences

    random.shuffle(head_pool)
    chunk_size = int(len(head_pool) / d_bound)
    level = 0

    #print(head_pool)

    for i, head in enumerate(head_pool):
        if i != 0 and level < d_bound-1 and i % chunk_size == 0: level += 1
        level_start = chunk_size*level

        # Don't want to contain any assumption twice
        selection_set = list(set(assumptions+head_pool[level_start:]))

        #print(i, level, level_start)
        #print(head, sorted(selection_set))

        n_rules_in_this_head = random.choice(n_rules_per_head)
        for _ in range(n_rules_in_this_head):
            size_of_body = random.choice(size_of_bodies)

            body = random.sample(assumptions+selection_set, size_of_body)
            rules.append((head, body))

    return assumptions, sentences, contraries, rules

def create_framework(n_sentences, n_assumptions, n_rules_per_head,
    size_of_bodies, cycle_prob, nonflat_coef):
    """
    Create a random framework.

    sentences contains the non-assumption sentences.
    n_rules_per_head should be a list exhausting the possible number of rules each head can have
    size_of_bodies should be a list exhausting the possible number of sentences in any rule body
    These should hold in order to get non-counterintuitive results:
    - n_assumptions < n_sentences
    - max(size_of_bodies) <= n_sentences+n_assumptions
    """

    assumptions = ["a" + str(i) for i in range(n_assumptions)]
    sentences = ["s" + str(i) for i in range(n_sentences-n_assumptions)]

    # contrary: Asms ---> Literals
    contraries = {asmpt: random.choice(sentences+assumptions) for asmpt in assumptions}

    # order sentences to avoid cycles
    random.shuffle(sentences)
    rules = []

    head_pool = None
    if nonflat_coef > 0:
        # certain percentage of the assumptions are also the heads of rules 
        head_pool = sentences+random.sample(assumptions, int(len(assumptions)*nonflat_coef))
        #head_pool = sentences+assumptions
    else:
        head_pool = sentences

    for i, head in enumerate(head_pool):
        # randomly choose the number of rules that have atom as head
        n_rules_in_this_head = random.choice(n_rules_per_head)
        for _ in range(n_rules_in_this_head):
            # randomly choose how many body elements this rule will have
            size_of_body = random.choice(size_of_bodies)

            # only allow stuff to occur in body if it is lower in the (topological) order
            n_available = len(assumptions) + i

            # include all the assumptions and the number of sentences already seen 
            selection_set = assumptions+sentences[:i]
            # add potentially cycle creating sentences to the selection set with a given probability
            extra_selection = random.sample(sentences[i:], min(len(sentences[i:]), int(cycle_prob*len(sentences))))
            selection_set.extend(extra_selection)

            #body = random.sample(assumptions+sentences[:i], min(size_of_body, n_available))
            body = random.sample(assumptions+selection_set, min(size_of_body, n_available))
            rules.append((head, body))

    return assumptions, sentences, contraries, rules

def create_atomic_abaf(n_sentences, n_assumptions, n_rules_per_head,
    size_of_bodies, nonflat_coef):

    assumptions = ["a" + str(i) for i in range(n_assumptions)]
    sentences = ["s" + str(i) for i in range(n_sentences-n_assumptions)]

    # contraries = {asmpt: random.choice(sentences+assumptions) for asmpt in assumptions}
    contraries = dict(zip(assumptions, sentences))

    rules = []

    head_pool = None
    if nonflat_coef > 0:
        head_pool = sentences+random.sample(assumptions, int(len(assumptions)*nonflat_coef))
        #head_pool = sentences+assumptions
    else:
        head_pool = sentences

    for i, head in enumerate(head_pool):
        n_rules_in_this_head = random.choice(n_rules_per_head)
        for _ in range(n_rules_in_this_head):
            size_of_body = random.choice(size_of_bodies)
            # rule body consists of assumptions only 
            body = random.sample(assumptions, min(size_of_body,len(assumptions)))
            rules.append((head, body))

    return assumptions, sentences, contraries, rules

def print_ASP(assumptions, contraries, rules, out_filename, query=None):
    """
    Print the given framework in ASP format.
    """
    with open(out_filename, 'w') as out:
        for asm in assumptions:
            out.write("assumption(" + asm + ").\n")
        for ctr in contraries:
            out.write("contrary(" + ctr + "," + contraries.get(ctr) + ").\n")
        for i, rule in enumerate(rules):
            out.write("head(" + str(i) + "," + rule[0] + ").\n")
            if rule[1]:
                for body in rule[1]:
                    out.write("body(" + str(i) + "," + body + ").\n")
        if query:
            out.write("query(" + query + ").")

def print_ABAF(num_elems, assumptions, contraries, rules, out_filename):
    with open(out_filename, 'w') as out:
        out.write(f"p aba {num_elems}\n")
        for asm in assumptions:
            out.write(f"a {asm}\n")
        for ctr in contraries:
            out.write(f"c {ctr} {contraries.get(ctr)}\n")
        for rule in rules:
            out.write(f"r {rule[0]} {' '.join(map(str, rule[1]))}\n")
    return 

'''
sen = int(sys.argv[1])
#sen = 3500
#n_a = int(round(0.6*sen))
#n_a = int(round(0.15*sen))
n_a = int(round(0.3*sen))
n_rph = range(1,3)
#n_rph = range(1,10)
#n_spb = range(1,10)
n_spb = range(1,3)
#n_rph = range(1,min(int(round(sen/8)),20))
#n_spb = range(1,min(int(round(sen/7)),20))
#n_rph = range(1,int(round(sen/30)))
#n_spb = range(1,int(round(sen/30)))
#n_rph = range(1,min(int(round(sen/8)),5))
#n_spb = range(1,int((1/15)*sen))
#n_spb = range(1,int(round(sen/7)))
#cycle_prob = 0
#cycle_prob = 0.06
#cycle_prob = float(sys.argv[2])

nonflat_coef = 0.5
#framework = create_framework(sen, n_a, n_rph, n_spb, cycle_prob, nonflat)
#print_ASP(framework[0], framework[2], framework[3], "acyclic_benchmark.asp", "s0")
#framework = create_atomic_abaf(sen, n_a, n_rph, n_spb, nonflat_coef)
#print_ASP(framework[0], framework[2], framework[3], "acyclic_atomic_benchmark.asp", "s0")

d_bound = 3
framework = create_depth_bounded_framework(sen, n_a, n_rph, n_spb, nonflat_coef, d_bound)
print_ASP(framework[0], framework[2], framework[3], "depth_bounded_benchmark.asp", "s0")

'''
def generate(asp_directory, aba_directory, identifier):
    nonflat = True

    # number of atoms in total
    sens = [25,50,75,100,250,500,750,1000,2000,3000,4000,5000]
    #sens = [100,160,220,280,340,400]
    # no of rules deriving any atom
    n_rules_max = [16, 8, 4, 2]
    # rule size (no of atoms in body)
    rule_size_max = [16, 8, 4, 2]
    # ratio of atoms that are assumptions
    asmpt_ratio = [0.25, 0.4, 0.5, 0.6, 0.75] 
    d_bounds = [3, 6]
    # ratio of assumptions occurring as rule heads. For flat instances, just make nonflat_coef 0
    # (e.g. by setting the list nonflat_coefs to [0])
    nonflat_coefs = [0]
    for sen in sens:
        for k in asmpt_ratio:
            for rph_max in n_rules_max:
                for spb_max in rule_size_max:
                    for c_prob in [0.01, 0.03, 0.05, 0.07, 0.09]: # 
                    # for d_bound in d_bounds:
                        #if c_prob == 0: continue    # NOTE: only for atomic and depth bounded!
                        for nonflat_coef in nonflat_coefs:
                            for i in range(10):
                                random.seed(BASE_SEED + i)
                                cycle_prob = c_prob
                                n_a = int(round(k*sen))
                                n_rph = range(1,rph_max+1)
                                n_spb = range(1,spb_max+1)

                                # For atomic and basic
                                asp_filename = f"{asp_directory}/{identifier}_s{sen}_c{cycle_prob}_n{nonflat_coef}_a{k}_r{rph_max}_b{spb_max}_{i}.asp"
                                aba_filename = f"{aba_directory}/{identifier}_s{sen}_c{cycle_prob}_n{nonflat_coef}_a{k}_r{rph_max}_b{spb_max}_{i}.aba"
                                # QUESTION: sentences same as atoms? 
                                # to make sure this makes sense, n_a < sen
                                # max(n_spb) <= n_sentences+n_assumptions
                                framework = create_framework(sen, n_a, n_rph, n_spb, cycle_prob, nonflat_coef)
                                print_ASP(framework[0], framework[2], framework[3], asp_filename)
                                print_ABAF(sen,framework[0], framework[2], framework[3], aba_filename) 
                                # print_ASP(framework[0], framework[2], framework[3], filename, "s0")

                                # For depth bounded
                                # filename = f"{directory}/{identifier}_s{sen}_d{d_bound}_n{nonflat_coef}_a{k}_r{rph_max}_b{spb_max}_{i}.asp"
                                # print(filename)
                                # #print_ASP(framework[0], framework[2], framework[3], filename, "s0")
                                # #atomic_framework = create_atomic_abaf(sen, n_a, n_rph, n_spb, nonflat_coef)
                                # #print_ASP(atomic_framework[0], atomic_framework[2], atomic_framework[3], filename, "s0")
                                # framework = create_depth_bounded_framework(sen, n_a, n_rph, n_spb, nonflat_coef, d_bound)
                                # print_ASP(framework[0], framework[2], framework[3], filename, "s0")

def generate_atomic(asp_directory, aba_directory, identifier):
    # number of atoms in total
    sens = [20,40,60]
    #sens = [100,160,220,280,340,400]
    # no of rules deriving any atom. UP TO :) x per atom.
    n_rules_max = [8, 4, 2]
    # rule size (no of atoms in body)
    rule_size_max = [16, 8, 4, 2]
    # ratio of atoms that are assumptions
    asmpt_ratio = [0.5] 
    # ratio of assumptions occurring as rule heads. For flat instances, just make nonflat_coef 0
    # (e.g. by setting the list nonflat_coefs to [0])
    nonflat_coefs = [0.01, 0.05, 0.1, 0.2]
    for sen in sens:
        for k in asmpt_ratio:
            for rph_max in n_rules_max:
                for spb_max in rule_size_max:
                    for nonflat_coef in nonflat_coefs:
                        for i in range(10):
                            random.seed(BASE_SEED + i)
                            n_a = int(round(k*sen))
                            n_rph = range(1,rph_max+1)
                            n_spb = range(1,spb_max+1)

                            # For atomic and basic
                            asp_filename = f"{asp_directory}/{identifier}_s{sen}_n{nonflat_coef}_a{k}_r{rph_max}_b{spb_max}_{i}.asp"
                            aba_filename = f"{aba_directory}/{identifier}_s{sen}_n{nonflat_coef}_a{k}_r{rph_max}_b{spb_max}_{i}.aba"
                            # QUESTION: sentences same as atoms? 
                            # to make sure this makes sense, n_a < sen
                            # max(n_spb) <= n_sentences+n_assumptions
                            framework = create_atomic_abaf(sen, n_a, n_rph, n_spb, nonflat_coef)
                            print_ASP(framework[0], framework[2], framework[3], asp_filename)
                            print_ABAF(sen,framework[0], framework[2], framework[3], aba_filename) 
                            # print_ASP(framework[0], framework[2], framework[3], filename, "s0")


if __name__ == '__main__':
    # set a new seed per run to generate new examples 
    # for the flat and non-flat instances we used seeds 42,43,44,45
    parser = argparse.ArgumentParser()
    parser.add_argument('-asp', '--aspdirectory')
    parser.add_argument('-aba', '--abadirectory')
    parser.add_argument('-i', '--identifier')
    args = parser.parse_args()

    asp_directory = args.aspdirectory
    aba_directory = args.abadirectory
    identifier = args.identifier
    
    ## set it manually for testing
    asp_directory = "asps/"
    aba_directory = "abaf/"
    identifier = "nf_atm"
    generate_atomic(asp_directory, aba_directory, identifier)