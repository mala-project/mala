import torch


class DataScaler:
    """Scales input and output data. Sort of emulates the functionality
    of the scikit-learn library, but by implementing the class by ourselves we have more freedom."""

    def __init__(self, typestring):
        self.typestring = typestring
        self.scale_standard = False
        self.scale_normal = False
        self.feature_wise = False
        self.parse_typestring()

        self.means = torch.empty(0)
        self.stds = torch.empty(0)
        self.maxs = torch.empty(0)
        self.mins = torch.empty(0)
        self.total_mean = torch.empty(0)
        self.total_std = torch.empty(0)
        self.total_max = torch.empty(0)
        self.total_min = torch.empty(0)
        self.cantransform = False

    def parse_typestring(self):
        self.scale_standard = False
        self.scale_normal = False
        self.feature_wise = False

        if "standard" in self.typestring:
            self.scale_standard = True
        if "normal" in self.typestring:
            self.scale_normal = True
        if "feature-wise" in self.typestring:
            self.feature_wise = True
        if self.scale_standard is False and self.scale_normal is False:
            print("No data rescaling will be performed.")
            return
        if self.scale_standard is True and self.scale_normal is True:
            raise Exception("Invalid input data rescaling.")

    def fit(self, unscaled):
        """Compute the quantities used for scaling."""
        if self.scale_standard is False and self.scale_normal is False:
            return
        else:
            with torch.no_grad():
                if self.feature_wise:

                    ##########################
                    # Feature-wise-scaling
                    ##########################

                    if self.scale_standard:
                        self.means = torch.mean(unscaled, 0, keepdim=True)
                        self.stds = torch.std(unscaled, 0, keepdim=True)

                    if self.scale_normal:
                        self.maxs = torch.max(unscaled, 0, keepdim=True)
                        self.mins = torch.min(unscaled, 0, keepdim=True)

                else:

                    ##########################
                    # Total scaling
                    ##########################

                    if self.scale_standard:
                        self.total_mean = torch.mean(unscaled)
                        self.total_std = torch.std(unscaled)

                    if self.scale_normal:
                        self.total_max = torch.max(unscaled)
                        self.total_min = torch.min(unscaled)

        self.cantransform = True

    def transform(self, unscaled):

        # First we need to find out if we even have to do anything.
        if self.scale_standard is False and self.scale_normal is False:
            return unscaled

        if self.cantransform is False:
            return unscaled

        # Perform the actual scaling, but use no_grad to make sure
        # that the next couple of iterations stay untracked.
        with torch.no_grad():
            if self.feature_wise:

                ##########################
                # Feature-wise-scaling
                ##########################

                if self.scale_standard:
                    unscaled = (unscaled - self.means) / self.stds
                    return unscaled

                if self.scale_normal:
                    unscaled = (unscaled - self.mins.values) / (self.maxs.values - self.mins.values)
                    return unscaled

            else:

                ##########################
                # Total scaling
                ##########################

                if self.scale_standard:
                    unscaled = (unscaled - self.total_mean) / self.total_std
                    return unscaled

                if self.scale_normal:
                    unscaled = (unscaled - self.total_min) / (self.total_max - self.total_min)
                    return unscaled

    def inverse_transform(self, scaled):

        # First we need to find out if we even have to do anything.
        if self.scale_standard is False and self.scale_normal is False:
            return scaled

        if self.cantransform is False:
            return scaled

        # Perform the actual scaling, but use no_grad to make sure
        # that the next couple of iterations stay untracked.
        with torch.no_grad():
            if self.feature_wise:

                ##########################
                # Feature-wise-scaling
                ##########################

                if self.scale_standard:
                    scaled = (scaled * self.stds) + self.means
                    return scaled

                if self.scale_normal:
                    scaled = (scaled*(self.maxs.values - self.mins.values)) + self.mins.values
                    return scaled

            else:

                ##########################
                # Total scaling
                ##########################

                if self.scale_standard:
                    scaled = (scaled * self.total_std) + self.total_mean
                    return scaled

                if self.scale_normal:
                    scaled = (scaled*(self.total_max - self.total_min)) + self.total_min
                    return scaled
