class TargetBase:
    """Base class for a target quantity parser. Target parsers read the target quantity
    (i.e. the quantity the NN will learn to predict) from a specified file format."""
    def __init__(self, p):
        self.parameters = p.targets
        self.target_length = 0

    def read_from_cube(self):
        raise Exception("No function defined to read this quantity from a .cube file.")
