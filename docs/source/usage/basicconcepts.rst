Basic concepts
==============

Workflow
*********

The principal idea behind MALA is to have easily reproducible workflows by
limiting the amount of code written to describe a workflow, and rather use
a single ``Parameters`` object to describe specifics.

Consider this piece of code:

      .. code-block:: python

          import mala

          parameters = mala.Parameters()
          datahandler = mala.DataHandler(parameters)
          datahandler.add_snapshot(...)
          network = mala.Network(parameters)
          trainer = mala.Trainer(parameters, network, datahandler)
          trainer.train_network()

Given that ``...`` contains meaningful information to an *atomic snapshot*
this piece of code will correctly load and scale data, instantiate a neural
network, and train it. What data is loaded, how it is scaled, what kind of
neural network will be used and how it's trained - all that is controlled
by the ``Parameters`` object. For most tasks there exist Interface-classes
that can be directly called, like ``Network`` or ``Descriptor``.
These will, under the hood, give you the correct object. However, it may
sometimes be useful to instead directly instantiate an object, e.g. for post-
processing:

      .. code-block:: python

          import mala

          parameters = mala.Parameters()
          ldos_calculator = mala.LDOS(parameters)

The Parameters object
**********************

The ``Parameters`` object lies at the heart of every MALA workflow.
It has several subobjects for its various tasks:

      .. code-block:: python

            parameters.data # How data is loaded
            parameters.descriptors # What type of descriptors are used/calculated (=input data)
            parameters.hyperparameters  # How are hyperparameters optimized
            parameters.network # What type of network is used
            parameters.running # How is the network run (either for training or inference)
            parameters.targets # What target is learned (=output data)

and in addition, a few properties used to direct resource management:

      .. code-block:: python

            parameters.use_gpu # Use a GPU via CUDA for network operations
            parameters.use_horovod # Use horovod for parallel training
            parameters.use_mpi # Use MPI for parallel inference

The values for the parameters can directly be written in your python file.
Often, it makes sense to instead load and save parameters to a file, and
do edits there. The default format is ``.json``, which enables direct
manipulation of parameters. For this, useful functions are:

      .. code-block:: python

            parameters.comment  # Attach a string to the parameters, to keep track
            parameters.save(...) # Save the Parameters to file
            parameters.show() # Print all values of current object
            new_parameters = mala.Parameters.parameters.load_from_file(...) # Create a new Parameters object from file
            parameters.manual_seed # Set a manual seed to compare network runs
            parameters.verbosity # Adjust the verbosity of MALA's output

Using optional modules
************************

You can use ``mala.check_modules()`` to find out which optional modules are
installed on your system. The output will look something like this:

      .. code-block:: none

            mpi4py: 	 installed 	 Enables inference parallelization.
            horovod: 	 not installed 	 Enables training parallelization.
            lammps: 	 installed 	 Enables descriptor calculation for data preprocessing and inference.
            oapackage: 	 installed 	 Enables usage of OAT method for hyperparameter optimization.
            pqkmeans: 	 installed 	 Enables clustering of training data.
            total_energy: 	 not installed 	 Enables calculation of total energy.

Attempting to use one of those functionalities without having the modules
installed will cause a crash.
