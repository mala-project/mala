"""DataScaler class for scaling DFT data."""
import pickle

try:
    import horovod.torch as hvd
except ModuleNotFoundError:
    # Warning is thrown by parameters class
    pass
import numpy as np
import torch

from mala.common.parameters import printout


class DataScaler:
    """Scales input and output data.

    Sort of emulates the functionality of the scikit-learn library, but by
    implementing the class by ourselves we have more freedom.

    Parameters
    ----------
    typestring :  string
        Specifies how scaling should be performed.
        Options:

        - "None": No normalization is applied.
        - "standard": Standardization (Scale to mean 0,
          standard deviation 1)
        - "normal": Min-Max scaling (Scale to be in range 0...1)
        - "feature-wise-standard": Row Standardization (Scale to mean 0,
          standard deviation 1)
        - "feature-wise-normal": Row Min-Max scaling (Scale to be in range
          0...1)

    use_horovod : bool
        If True, the DataScaler will use horovod to check that data is
        only saved on the root process in parallel execution.
    """

    def __init__(self, typestring, use_horovod=False):
        self.use_horovod = use_horovod
        self.typestring = typestring
        self.scale_standard = False
        self.scale_normal = False
        self.feature_wise = False
        self.cantransform = False
        self.__parse_typestring()

        self.means = torch.empty(0)
        self.stds = torch.empty(0)
        self.maxs = torch.empty(0)
        self.mins = torch.empty(0)
        self.total_mean = torch.tensor(0)
        self.total_std = torch.tensor(0)
        self.total_max = torch.tensor(float('-inf'))
        self.total_min = torch.tensor(float('inf'))

        self.total_data_count = 0

    def __parse_typestring(self):
        """Parse the typestring to class attributes."""
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
            printout("No data rescaling will be performed.", min_verbosity=1)
            self.cantransform = True
            return
        if self.scale_standard is True and self.scale_normal is True:
            raise Exception("Invalid input data rescaling.")

    def start_incremental_fitting(self):
        """
        Start the incremental calculation of scaling parameters.

        This is necessary for lazy loading.
        """
        self.total_data_count = 0

    def incremental_fit(self, unscaled):
        """
        Add data to the incremental calculation of scaling parameters.

        This is necessary for lazy loading.

        Parameters
        ----------
        unscaled : torch.Tensor
            Data that is to be added to the fit.

        """
        if self.scale_standard is False and self.scale_normal is False:
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
                            self.means = \
                                self.total_data_count /\
                                (self.total_data_count + current_data_count) \
                                * old_mean + current_data_count / \
                                (self.total_data_count + current_data_count)\
                                * new_mean
                        else:
                            self.means = new_mean
                        if list(self.stds.size())[0] > 0:
                            self.stds = \
                                self.total_data_count / \
                                (self.total_data_count + current_data_count) \
                                * old_std ** 2 + current_data_count / \
                                (self.total_data_count + current_data_count) *\
                                new_std ** 2 + \
                                (self.total_data_count * current_data_count)\
                                / (self.total_data_count + current_data_count)\
                                ** 2 * (old_mean - new_mean) ** 2

                            self.stds = torch.sqrt(self.stds)
                        else:
                            self.stds = new_std
                        self.total_data_count += current_data_count

                    if self.scale_normal:
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
                        current_data_count = list(unscaled.size())[0]\
                                             * list(unscaled.size())[1]

                        new_mean = torch.mean(unscaled)
                        new_std = torch.std(unscaled)

                        old_mean = self.total_mean
                        old_std = self.total_std

                        self.total_mean = \
                            self.total_data_count / \
                            (self.total_data_count + current_data_count) * \
                            old_mean + current_data_count / \
                            (self.total_data_count + current_data_count) *\
                            new_mean

                        # This equation is taken from the Sandia code. It
                        # presumably works, but it gets slighly different
                        # results.
                        # Maybe we should check it at some point .
                        # I think it is merely an issue of numerical accuracy.
                        self.total_std = \
                            self.total_data_count / \
                            (self.total_data_count + current_data_count) * \
                            old_std ** 2 + \
                            current_data_count / \
                            (self.total_data_count + current_data_count) \
                            * new_std ** 2 + \
                            (self.total_data_count * current_data_count) / \
                            (self.total_data_count + current_data_count) \
                            ** 2 * (old_mean - new_mean) ** 2

                        self.total_std = torch.sqrt(self.total_std)
                        self.total_data_count += current_data_count

                    if self.scale_normal:
                        new_max = torch.max(unscaled)
                        if new_max > self.total_max:
                            self.total_max = new_max

                        new_min = torch.min(unscaled)
                        if new_min < self.total_min:
                            self.total_min = new_min

    def finish_incremental_fitting(self):
        """
        Indicate that all data has been added to the incremental calculation.

        This is necessary for lazy loading.
        """
        self.cantransform = True

    def fit(self, unscaled):
        """
        Compute the quantities necessary for scaling.

        Parameters
        ----------
        unscaled : torch.Tensor
            Data that on which the scaling will be calculated.

        """
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
                        self.maxs = torch.max(unscaled, 0, keepdim=True).values
                        self.mins = torch.min(unscaled, 0, keepdim=True).values

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
        """
        Transform data from unscaled to scaled.

        Unscaled means real world data, scaled means data as is used in
        the network. Data is transformed in-place.

        Parameters
        ----------
        unscaled : torch.Tensor
            Real world data.

        Returns
        -------
        scaled : torch.Tensor
            Scaled data.
        """
        # First we need to find out if we even have to do anything.
        if self.scale_standard is False and self.scale_normal is False:
            pass

        elif self.cantransform is False:
            raise Exception("Transformation cannot be done, this DataScaler "
                            "was never initialized")

        # Perform the actual scaling, but use no_grad to make sure
        # that the next couple of iterations stay untracked.
        with torch.no_grad():
            if self.feature_wise:

                ##########################
                # Feature-wise-scaling
                ##########################

                if self.scale_standard:
                    unscaled -= self.means
                    unscaled /= self.stds

                if self.scale_normal:
                    unscaled -= self.mins
                    unscaled /= (self.maxs - self.mins)

            else:

                ##########################
                # Total scaling
                ##########################

                if self.scale_standard:
                    unscaled -= self.total_mean
                    unscaled /= self.total_std

                if self.scale_normal:
                    unscaled -= self.total_min
                    unscaled /= (self.total_max - self.total_min)

    def inverse_transform(self, scaled, as_numpy=False):
        """
        Transform data from scaled to unscaled.

        Unscaled means real world data, scaled means data as is used in
        the network.

        Parameters
        ----------
        scaled : torch.Tensor
            Scaled data.

        as_numpy : bool
            If True, a numpy array is returned, otherwsie.

        Returns
        -------
        unscaled : torch.Tensor
            Real world data.

        """
        # First we need to find out if we even have to do anything.
        if self.scale_standard is False and self.scale_normal is False:
            unscaled = scaled

        else:
            if self.cantransform is False:
                raise Exception("Backtransformation cannot be done, this "
                                "DataScaler was never initialized")

            # Perform the actual scaling, but use no_grad to make sure
            # that the next couple of iterations stay untracked.
            with torch.no_grad():
                if self.feature_wise:

                    ##########################
                    # Feature-wise-scaling
                    ##########################

                    if self.scale_standard:
                        unscaled = (scaled * self.stds) + self.means

                    if self.scale_normal:
                        unscaled = (scaled*(self.maxs
                                            - self.mins)) + self.mins

                else:

                    ##########################
                    # Total scaling
                    ##########################

                    if self.scale_standard:
                        unscaled = (scaled * self.total_std) + self.total_mean

                    if self.scale_normal:
                        unscaled = (scaled*(self.total_max
                                            - self.total_min)) + self.total_min
#
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
        # If we use horovod, only save the network on root.
        if self.use_horovod:
            if hvd.rank() != 0:
                return
        if save_format == "pickle":
            with open(filename, 'wb') as handle:
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
                loaded_scaler = pickle.load(open(file, 'rb'))
            else:
                loaded_scaler = pickle.load(file)
        else:
            raise Exception("Unsupported parameter save format.")

        return loaded_scaler
