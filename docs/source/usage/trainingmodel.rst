Training a model
=================

Neural networks are powerful machine learning tools in principle capable of
approximating any function. In MALA, neural networks are built using PyTorch.
Hyperparameter optimization can be done using optuna and custom routines.
To see how a model is trained, refer to e.g. ``ex01_singleshot``.

Optimal performance
*******************

MALA by default uses sub-optimal training parameters to guarantee compatibility
with a wide range of systems. Users are encouraged to activate advanced
features for optimal performance. The following is a list of parameters that
can and should be looked into.

    .. code-block:: python

        parameters = mala.Parameters()
        """
        This setting is pretty straightforward - wherever possible, use a GPU.
        """
        parameters.use_gpu = False # True: Use GPU
        """
        Multiple workers allow for faster data processing, but require
        additional CPU/RAM power. A good setup is e.g. using 4 CPUs attached
        to one GPU and setting the num_workers to 4.
        Please be aware that using multiple workers is currently not supported
        with horovod.
        """
        parameters.running.num_workers = 0 # set to e.g. 4
        """
        MALA supports a faster implementation of the TensorDataSet class
        from the torch library. Turning it on will drastically improve
        performance.
        """
        parameters.data.use_fast_tensor_data_set = False # True: Faster data loading
        """
        Likewise, using CUDA graphs improve performance by optimizing GPU
        usage. Be careful, this option is only availabe from CUDA 11.0 onwards.
        """
        parameters.running.use_graphs = False # True: Better GPU utilization
        """
        Using mixed precision can also improve performance, but only if
        the model is large enough.
        """
        parameters.running.use_mixed_precision = False # True: Improved performance for large models
