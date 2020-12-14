import torch

class ScalingOptions:
    """Helper class to represent scaling options."""
    def __init__(self, scale_standard, scale_normal, feature_wise):
        self.scale_standard = scale_standard
        self.scale_normal = scale_normal
        self.feature_wise = feature_wise

class DataScaler:
    """Scales input and output data."""

    def __init__(self, p):
        self.parameters = p.scaling
        self.input_scaling = ScalingOptions(False, False, False)
        self.output_scaling = ScalingOptions(False, False, False)
        self.parse_parameters()

    def parse_parameters(self):
        ####################
        # Inputs.
        ####################

        self.input_scaling.scale_standard = False
        self.input_scaling.scale_normal = False
        self.input_scaling.feature_wise = False

        if "standard" in self.parameters.input_rescaling_type:
            self.input_scaling.scale_standard = True
        if "normal" in self.parameters.input_rescaling_type:
            self.input_scaling.scale_normal = True
        if "feature-wise" in self.parameters.input_rescaling_type:
            self.input_scaling.feature_wise = True
        if self.input_scaling.scale_standard is False and self.input_scaling.scale_normal is False:
            print("No input data rescaling will be performed.")
            return
        if self.input_scaling.scale_standard is True and self.input_scaling.scale_normal is True:
            raise Exception("Invalid input data rescaling.")

        ####################
        # Outputs.
        ####################

        self.output_scaling.scale_standard = False
        self.output_scaling.scale_normal = False
        self.output_scaling.feature_wise = False

        if "standard" in self.parameters.output_rescaling_type:
            self.output_scaling.scale_standard = True
        if "normal" in self.parameters.output_rescaling_type:
            self.output_scaling.scale_normal = True
        if "feature-wise" in self.parameters.output_rescaling_type:
            self.output_scaling.feature_wise = True
        if self.output_scaling.scale_standard is False and self.output_scaling.scale_normal is False:
            print("No output data rescaling will be performed.")
            return
        if self.output_scaling.scale_standard is True and self.output_scaling.scale_normal is True:
            raise Exception("Invalid output data rescaling.")

    def scale_input_tensor(self, intensor):
        """Scales a tensor using the parameters specified for input tensors."""
        self.__scale_tensor(intensor, self.input_scaling)

    def scale_output_tensor(self, outtensor):
        """Scales a tensor using the parameters specified for input tensors."""
        self.__scale_tensor(outtensor, self.output_scaling)

    def __scale_tensor(self, unscaled, scaling_options):

        # First we need to find out if we even have to do anything.
        if scaling_options.scale_standard is False and scaling_options.scale_normal is False:
            return

        # Perform the actual scaling, but use no_grad to make sure
        # that the next couple of iterations stay untracked.
        with torch.no_grad():
            if scaling_options.feature_wise:

                ##########################
                # Feature-wise-scaling
                ##########################

                if scaling_options.scale_standard:
                    means = torch.mean(unscaled, 0, keepdim=True)
                    stds = torch.std(unscaled, 0, keepdim=True)
                    unscaled = (unscaled - means) / stds

                if scaling_options.scale_normal:
                    maxs = torch.max(unscaled, 0, keepdim=True)
                    mins = torch.min(unscaled, 0, keepdim=True)
                    unscaled = (unscaled - mins.values) / (maxs.values-mins.values)

            else:

                ##########################
                # Total scaling
                ##########################

                if scaling_options.scale_standard:
                    single_mean = torch.mean(unscaled)
                    single_std = torch.std(unscaled)
                    unscaled = (unscaled - single_mean) / single_std

                if scaling_options.scale_normal:
                    single_max = torch.max(unscaled)
                    single_min = torch.min(unscaled)
                    unscaled = (unscaled - single_min) / (single_max - single_min)
