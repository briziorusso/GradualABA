class Rule:
    
    _used_identifiers = set() 
    _counter = 1 

    def __init__(self, head, body=None, name=None):
        if not head:
            raise ValueError("Head must be specified.")

        if name is None:
            name = f"r{Rule._counter}"
            Rule._counter += 1

        if name in Rule._used_identifiers:
            raise ValueError(f"Rule identifier '{name}' already exists. Choose a unique name.")

        self.name = name
        self.head = head 
        self.body = body if body else set()

        # Register the identifier as used
        Rule._used_identifiers.add(name)

    def __repr__(self):
        body_str = ", ".join(self.body) if self.body else ""
        if body_str:
            return f"{self.name}: {self.head} :- {body_str}."
        else:
            return f"{self.name}: {self.head}."

    @classmethod
    def reset_identifiers(cls):
        """Reset the used identifiers (for testing or reloading purposes)."""
        cls._used_identifiers.clear()
        cls._counter = 1  # Reset counter for fresh rule names
