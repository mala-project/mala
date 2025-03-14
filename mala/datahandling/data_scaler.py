"""DataScaler class for scaling DFT data."""

import pickle
import numpy as np
import torch
import torch.distributed as dist

from mala.common.parameters import printout
from mala.common.parallelizer import parallel_warn


# IMPORTANT: If you change the docstrings, make sure to also change them
# in the ParametersData subclass, because users do usually not interact
# with this class directly.
class DataScaler:
    """Scales input and output data.

    Sort of emulates the functionality of the scikit-learn library, but by
    implementing the class by ourselves we have more freedom. Specifically
    assumes data of the form (d,f), where d=x*y*z, i.e., the product of spatial
    dimensions, and f is the feature dimension.

    Parameters
    ----------
    typestring :  string
        Specifies how scaling should be performed.
        Options:

        - "None": No scaling is applied.
        - "standard": Standardization (Scale to mean 0,
          standard deviation 1) is applied to the entire array.
        - "minmax": Min-Max scaling (Scale to be in range 0...1) is applied
          to the entire array.
        - "feature-wise-standard": Standardization (Scale to mean 0,
          standard deviation 1) is applied to each feature dimension
          individually.
          I.e., if your training data has dimensions (d,f), then each
          of the f columns with d entries is scaled indiviually.
        - "feature-wise-minmax": Min-Max scaling (Scale to be in range
          0...1) is applied to each feature dimension individually.
          I.e., if your training data has dimensions (d,f), then each
          of the f columns with d entries is scaled indiviually.
        - "normal": (DEPRECATED) Old name for "minmax".
        - "feature-wise-normal": (DEPRECATED) Old name for
          "feature-wise-minmax"

    use_ddp : bool
        If True, the DataScaler will use ddp to check that data is
        only saved on the root process in parallel execution.

    Attributes
    ----------
    cantransform : bool
        If True, this scaler is set up to perform scaling.

    feature_wise : bool
        (Managed internally, not set to private due to legacy issues)

    maxs : torch.Tensor
        (Managed internally, not set to private due to legacy issues)

    means : torch.Tensor
        (Managed internally, not set to private due to legacy issues)

    mins : torch.Tensor
        (Managed internally, not set to private due to legacy issues)

    scale_minmax : bool
        (Managed internally, not set to private due to legacy issues)

    scale_standard : bool
        (Managed internally, not set to private due to legacy issues)

    stds : torch.Tensor
        (Managed internally, not set to private due to legacy issues)

    total_data_count : int
        (Managed internally, not set to private due to legacy issues)

    total_max : float
        (Managed internally, not set to private due to legacy issues)

    total_mean : float
        (Managed internally, not set to private due to legacy issues)

    total_min : float
        (Managed internally, not set to private due to legacy issues)

    total_std : float
        (Managed internally, not set to private due to legacy issues)

    typestring : str
        (Managed internally, not set to private due to legacy issues)

    use_ddp : bool
        (Managed internally, not set to private due to legacy issues)
    """

    def __init__(self, typestring, use_ddp=False):
        self.use_ddp = use_ddp
        self.typestring = typestring
        self.scale_standard = False
        self.scale_minmax = False
        self.feature_wise = False
        self.cantransform = False
        self.__parse_typestring()

        self.means = torch.empty(0)
        self.stds = torch.empty(0)
        self.maxs = torch.empty(0)
        self.mins = torch.empty(0)
        self.total_mean = torch.tensor(0)
        self.total_std = torch.tensor(0)
        self.total_max = torch.tensor(float("-inf"))
        self.total_min = torch.tensor(float("inf"))

        self.total_data_count = 0

    def __parse_typestring(self):
        """Parse the typestring to class attributes."""
        self.scale_standard = False
        self.scale_minmax = False
        self.feature_wise = False

        if "standard" in self.typestring:
            self.scale_standard = True
        if "normal" in self.typestring:
            parallel_warn(
                "Options 'normal' and 'feature-wise-normal' will be "
                "deprecated, starting in MALA v1.4.0. Please use 'minmax' and "
                "'feature-wise-minmax' instead.",
                min_verbosity=0,
                category=FutureWarning,
            )
            self.scale_minmax = True
        if "minmax" in self.typestring:
            self.scale_minmax = True
        if "feature-wise" in self.typestring:
            self.feature_wise = True
        if self.scale_standard is False and self.scale_minmax is False:
            printout("No data rescaling will be performed.", min_verbosity=1)
            self.cantransform = True
            return
        if self.scale_standard is True and self.scale_minmax is True:
            raise Exception("Invalid input data rescaling.")

    def reset(self):
        """
        Start the incremental calculation of scaling parameters.

        This is necessary for lazy loading.
        """
        self.total_data_count = 0

    def partial_fit(self, unscaled):
        """
        Add data to the incremental calculation of scaling parameters.

        This is necessary for lazy loading.

        Parameters
        ----------
        unscaled : torch.Tensor
            Data that is to be added to the fit.

        """
        if len(unscaled.size()) != 2:
            raise ValueError(
                "MALA DataScaler are designed for 2D-arrays, "
                "while a {0}D-array has been provided.".format(
                    len(unscaled.size())
                )
            )

        if self.scale_standard is False and self.scale_minmax is False:
            self.cantransform = True
            return
        else:
            with torch.no_grad():
                if self.feature_wise:

                    ##########################
                    # Feature-wise-scaling
                    ##########################

                    if self.scale_standard:
                        new_mean = torch.mean(unscaled, 0, keepdim=True)
                        new_std = torch.std(unscaled, 0, keepdim=True)

                        current_data_count = list(unscaled.size())[0]

                        old_mean = self.means
                        old_std = self.stds

                        if list(self.means.size())[0] > 0:
                            self.means = (
                                self.total_data_count
                                / (self.total_data_count + current_data_count)
                                * old_mean
                                + current_data_count
                                / (self.total_data_count + current_data_count)
                                * new_mean
                            )
                        else:
                            self.means = new_mean
                        if list(self.stds.size())[0] > 0:
                            self.stds = (
                                self.total_data_count
                                / (self.total_data_count + current_data_count)
                                * old_std**2
                                + current_data_count
                                / (self.total_data_count + current_data_count)
                                * new_std**2
                                + (self.total_data_count * current_data_count)
                                / (self.total_data_count + current_data_count)
                                ** 2
                                * (old_mean - new_mean) ** 2
                            )

                            self.stds = torch.sqrt(self.stds)
                        else:
                            self.stds = new_std
                        self.total_data_count += current_data_count

                    if self.scale_minmax:
                        new_maxs = torch.max(unscaled, 0, keepdim=True)
                        if list(self.maxs.size())[0] > 0:
                            for i in range(list(new_maxs.values.size())[1]):
                                if new_maxs.values[0, i] > self.maxs[i]:
                                    self.maxs[i] = new_maxs.values[0, i]
                        else:
                            self.maxs = new_maxs.values[0, :]

                        new_mins = torch.min(unscaled, 0, keepdim=True)
                        if list(self.mins.size())[0] > 0:
                            for i in range(list(new_mins.values.size())[1]):
                                if new_mins.values[0, i] < self.mins[i]:
                                    self.mins[i] = new_mins.values[0, i]
                        else:
                            self.mins = new_mins.values[0, :]

                else:

                    ##########################
                    # Total scaling
                    ##########################

                    if self.scale_standard:
                        current_data_count = (
                            list(unscaled.size())[0] * list(unscaled.size())[1]
                        )

                        new_mean = torch.mean(unscaled)
                        new_std = torch.std(unscaled)

                        old_mean = self.total_mean
                        old_std = self.total_std

                        self.total_mean = (
                            self.total_data_count
                            / (self.total_data_count + current_data_count)
                            * old_mean
                            + current_data_count
                            / (self.total_data_count + current_data_count)
                            * new_mean
                        )

                        # This equation is taken from the Sandia code. It
                        # presumably works, but it gets slighly different
                        # results.
                        # Maybe we should check it at some point .
                        # I think it is merely an issue of numerical accuracy.
                        self.total_std = (
                            self.total_data_count
                            / (self.total_data_count + current_data_count)
                            * old_std**2
                            + current_data_count
                            / (self.total_data_count + current_data_count)
                            * new_std**2
                            + (self.total_data_count * current_data_count)
                            / (self.total_data_count + current_data_count) ** 2
                            * (old_mean - new_mean) ** 2
                        )

                        self.total_std = torch.sqrt(self.total_std)
                        self.total_data_count += current_data_count

                    if self.scale_minmax:
                        new_max = torch.max(unscaled)
                        if new_max > self.total_max:
                            self.total_max = new_max

                        new_min = torch.min(unscaled)
                        if new_min < self.total_min:
                            self.total_min = new_min
        self.cantransform = True

    def fit(self, unscaled):
        """
        Compute the quantities necessary for scaling.

        Parameters
        ----------
        unscaled : torch.Tensor
            Data that on which the scaling will be calculated.

        """
        if len(unscaled.size()) != 2:
            raise ValueError(
                "MALA DataScaler are designed for 2D-arrays, "
                "while a {0}D-array has been provided.".format(
                    len(unscaled.size())
                )
            )

        if self.scale_standard is False and self.scale_minmax is False:
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

                    if self.scale_minmax:
                        self.maxs = torch.max(unscaled, 0, keepdim=True).values
                        self.mins = torch.min(unscaled, 0, keepdim=True).values

                else:

                    ##########################
                    # Total scaling
                    ##########################

                    if self.scale_standard:
                        self.total_mean = torch.mean(unscaled)
                        self.total_std = torch.std(unscaled)

                    if self.scale_minmax:
                        self.total_max = torch.max(unscaled)
                        self.total_min = torch.min(unscaled)

        self.cantransform = True

    def transform(self, unscaled, copy=False):
        """
        Transform data from unscaled to scaled.

        Unscaled means real world data, scaled means data as is used in
        the network. Data is transformed in-place.

        Parameters
        ----------
        unscaled : torch.Tensor
            Real world data.

        copy : bool
            If False, data is modified in-place. If True, a copy of the
            data is modified. Default is False.

        Returns
        -------
        scaled : torch.Tensor
            Scaled data.
        """
        if len(unscaled.size()) != 2:
            raise ValueError(
                "MALA DataScaler are designed for 2D-arrays, "
                "while a {0}D-array has been provided.".format(
                    len(unscaled.size())
                )
            )

        # Backward compatability.
        if not hasattr(self, "scale_minmax") and hasattr(self, "scale_normal"):
            self.scale_minmax = self.scale_normal

        # First we need to find out if we even have to do anything.
        if self.scale_standard is False and self.scale_minmax is False:
            pass

        elif self.cantransform is False:
            raise Exception(
                "Transformation cannot be done, this DataScaler "
                "was never initialized"
            )

        # Perform the actual scaling, but use no_grad to make sure
        # that the next couple of iterations stay untracked.
        scaled = unscaled.clone() if copy else unscaled

        with torch.no_grad():
            if self.feature_wise:

                ##########################
                # Feature-wise-scaling
                ##########################

                if self.scale_standard:
                    scaled -= self.means
                    scaled /= self.stds

                if self.scale_minmax:
                    scaled -= self.mins
                    scaled /= self.maxs - self.mins

            else:

                ##########################
                # Total scaling
                ##########################

                if self.scale_standard:
                    scaled -= self.total_mean
                    scaled /= self.total_std

                if self.scale_minmax:
                    scaled -= self.total_min
                    scaled /= self.total_max - self.total_min

        return scaled

    def inverse_transform(self, scaled, copy=False, as_numpy=False):
        """
        Transform data from scaled to unscaled.

        Unscaled means real world data, scaled means data as is used in
        the network.

        Parameters
        ----------
        scaled : torch.Tensor
            Scaled data.

        as_numpy : bool
            If True, a numpy array is returned, otherwise a torch tensor.

        copy : bool
            If False, data is modified in-place. If True, a copy of the
            data is modified. Default is False.

        Returns
        -------
        unscaled : torch.Tensor
            Real world data.

        """
        if len(scaled.size()) != 2:
            raise ValueError(
                "MALA DataScaler are designed for 2D-arrays, "
                "while a {0}D-array has been provided.".format(
                    len(scaled.size())
                )
            )

        # Backward compatability.
        if not hasattr(self, "scale_minmax") and hasattr(self, "scale_normal"):
            self.scale_minmax = self.scale_normal

        # Perform the actual scaling, but use no_grad to make sure
        # that the next couple of iterations stay untracked.
        unscaled = scaled.clone() if copy else scaled

        # First we need to find out if we even have to do anything.
        if self.scale_standard is False and self.scale_minmax is False:
            pass

        else:
            if self.cantransform is False:
                raise Exception(
                    "Backtransformation cannot be done, this "
                    "DataScaler was never initialized"
                )

            # Perform the actual scaling, but use no_grad to make sure
            # that the next couple of iterations stay untracked.
            with torch.no_grad():
                if self.feature_wise:

                    ##########################
                    # Feature-wise-scaling
                    ##########################

                    if self.scale_standard:
                        unscaled *= self.stds
                        unscaled += self.means

                    if self.scale_minmax:
                        unscaled *= self.maxs - self.mins
                        unscaled += self.mins

                else:

                    ##########################
                    # Total scaling
                    ##########################

                    if self.scale_standard:
                        unscaled *= self.total_std
                        unscaled += self.total_mean

                    if self.scale_minmax:
                        unscaled *= self.total_max - self.total_min
                        unscaled += self.total_min

        if as_numpy:
            return unscaled.detach().numpy().astype(np.float64)
        else:
            return unscaled

    def save(self, filename, save_format="pickle"):
        """
        Save the Scaler object so that it can be accessed again later.

        Parameters
        ----------
        filename : string
            File in which the parameters will be saved.

        save_format :
            File format which will be used for saving.
        """
        # If we use ddp, only save the network on root.
        if self.use_ddp:
            if dist.get_rank() != 0:
                return
        if save_format == "pickle":
            with open(filename, "wb") as handle:
                pickle.dump(self, handle, protocol=4)
        else:
            raise Exception("Unsupported parameter save format.")

    @classmethod
    def load_from_file(cls, file, save_format="pickle"):
        """
        Load a saved Scaler object.

        Parameters
        ----------
        file : string or ZipExtFile
            File from which the parameters will be read.

        save_format :
            File format which was used for saving.

        Returns
        -------
        data_scaler : DataScaler
            DataScaler which was read from the file.
        """
        if save_format == "pickle":
            if isinstance(file, str):
                loaded_scaler = pickle.load(open(file, "rb"))
            else:
                loaded_scaler = pickle.load(file)
        else:
            raise Exception("Unsupported parameter save format.")

        return loaded_scaler
