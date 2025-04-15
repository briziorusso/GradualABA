from .Argument import Argument

def remove_duplicates(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result

def sort_by_key(lst, order):
    return sorted(lst, key=lambda x: order.index(x))


# update to class BSAF. this is still very incomplete. 

class BSAF:
    def __init__(self, arguments=None):
        # arguments: [a,b,c]
        self.arguments = remove_duplicates(arguments) if arguments is not None else []
        # we do not accept attacks and supports on this stage
        # self.setAttacks = setAttacks if setAttacks is not None else {}
        # self.setSupports = setSupports if setSupports is not None else {}

        # setAttacks and setSupports: {a: [[1,0,1],[0,1,1]], b: [0,1,1]], c: [[1,1,0]]}
        self.setAttacks = {self.arguments[i]: [] for i in range(len(self.arguments))}
        self.setSupports = {self.arguments[i]: [] for i in range(len(self.arguments))}

    def add_argument(self, argument):
        if not isinstance(argument, Argument):
            raise TypeError("argument must be of type Argument")
        
        if argument in self.arguments:
            # do not add duplicate arguments
            return
        else:   
            self.arguments.append(argument)
            self.setAttacks[argument] = []
            self.setSupports[argument] = []

    def add_attack(self, attacker, attacked):
        # expect: attacker = [b,c], attacked = a

        if not isinstance(attacked, Argument):
            raise TypeError("attacked must be of type Argument")

        if not isinstance(attacker, list):
            raise TypeError("attacker must be a list of Arguments")
        
        for arg in attacker:
            if not isinstance(arg, Argument):
                raise TypeError("each attacker in list must be of type Argument")
            
            if arg not in self.arguments:
                self.add_argument(arg)
        
        if attacked not in self.arguments:
            self.add_argument(attacked)

        # sort attacker by order of arguments
        attacker_sorted = sort_by_key(attacker, self.arguments)
        # check if attacker_sorted is in setAttacks of attacked
        if attacker_sorted not in self.setAttacks[attacked]:
            self.setAttacks[attacked].append(attacker_sorted)

    def add_support(self, supporter, supported):

        if not isinstance(supported, Argument):
            raise TypeError("supported must be of type Argument")
        if not isinstance(supporter, list):
            raise TypeError("supporter must be a list of Arguments")    
        for arg in supporter: 
            if not isinstance(arg, Argument):
                raise TypeError("each supporter in list must be of type Argument")
            
            if arg not in self.arguments:
                self.add_argument(arg)
        if supported not in self.arguments:
            self.add_argument(supported)
            
        supporter_sorted = sort_by_key(supporter, self.arguments)
        if supporter_sorted not in self.setAttacks[supported]:
            self.setSupports[supported].append(supporter_sorted)
    


    def __repr__(self) -> str:
        return f"BSAF:\nArguments: {self.arguments}\nAttacks: {self.setAttacks}\nSupports: {self.setSupports}"
    
    def __str__(self):
        attacks = []
        supports = []
        for argument in self.arguments:
            for setattack in self.setAttacks[argument]:
                tmp = ",".join([arg.name for arg in setattack])
                tmp2 = "({"+tmp + "},"+argument.name+")"
                attacks.append(tmp2)

            for set in self.setSupports[argument]:
                tmp = ",".join([arg.name for arg in set])
                tmp2 = "({"+tmp + "},"+argument.name+")"
                supports.append(tmp2)

        args = ",".join([args.name for args in self.arguments])
        atts = ",".join(attacks)
        supps = ",".join(supports)
        return f"BSAF: Arguments: {args}\nAttacks: {atts}\nSupports: {supps}"