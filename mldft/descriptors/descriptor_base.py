class DescriptorBase():
    def __init__(self, p):
        """Base class for a descriptor. Descriptors transform DFT data into usable
        inputs for our neural network."""
        self.parameters = p.descriptors
        self.fingerprint_length = -1  # so iterations will fail
        self.dbg_grid_dimensions = p.debug.grid_dimensions
