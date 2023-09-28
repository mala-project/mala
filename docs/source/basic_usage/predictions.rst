Using ML-DFT models for predictions
===================================

Having trained and saved a model, this model can now be used for predictions.
This guide follows the examples ``ex05_run_predictions.py`` and
``ex06_ase_calculator.py``. In the :ref:`advanced section <production>` on
this topic, performance tweaks and extended access to observables are covered.

In order to get direct access to electronic structure via ML, MALA uses
the ``Predictor`` class. Provided that the trained model was saved with
all the necessary information on the bispectrum descriptors and the LDOS,
no further parameters have to be set here. You have to load the
model and acquire atomic positions (e.g., from a previous molecular dynamics
simulation or other source) via

      .. code-block:: python

            import ase

            parameters, network, data_handler, predictor = mala.Predictor.\
                load_run("be_model")
            atoms = ase.io.read(...)

Please note that MALA interfaces are `ASE <https://wiki.fysik.dtu.dk/ase/>`_
compatible. ASE is a powerful library for atomistic modeling in Python,
providing functionalities such as a format for atomic structures and
interfaces to simulation software. Having such an atomic configuration
in the form of a ``ase.Atoms`` object, a prediction can
be made via

      .. code-block:: python

            ldos = predictor.predict_for_atoms(atoms)

The resulting LDOS can then be processed via the calculator object of
the predictor class to access various observables of interest, such as

      .. code-block:: python

            ldos_calculator: mala.LDOS = predictor.target_calculator
            ldos_calculator.read_from_array(ldos)
            # Total energy of the system
            ldos_calculator.total_energy
            # Electronic density
            ldos_calculator.density
            # Electronic density of states
            ldos_calculator.density_of_states

Please note that in order to calculate the total energy, you have to
provide a pseudopotential (specifically the pseudopotential used during
data generation) via ``parameters.targets.pseudopotential_path = ...``.

Using the MALA ASE calculator
*****************************

For even easier integration into materials modelling workflows, MALA
provides an ASE calculator interface, called ``mala.MALA``
. An ASE calculator is an interface
to an electronic structure simulation code with a specified API;
calculators exist for many simulation codes, such as Quantum ESPRESSO,
GPAW, etc.

The merit of using the MALA ASE calculator is that existing workflows
do not have to be altered - they can be performed faster with
the power of ML. Usage is fairly simple, first, a model is loaded as an
ASE calculator and set a pseudopotential path

      .. code-block:: python

            calculator = mala.MALA.load_model("be_model")
            calculator.mala_parameters.targets.pseudopotential_path = ...

Please note that the ``mala_parameters`` property is the ``Parameters``
object with which the calculator was created/loaded and can be used
just as a ``Parameters`` object to adjust inference settings
. It is not called ``parameters`` since that keyword is already
in use by ASE itself.

Afterwards, define/load an ASE atoms object, set the ``MALA`` object
as a calculator and use the ASE interface to perform the calculation.

      .. code-block:: python

            atoms = read(os.path.join(data_path, "Be_snapshot1.out"))
            atoms.set_calculator(calculator)
            atoms.get_potential_energy()

For more information on ASE calculators, also see the official
`ASE documentation <https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html>`_.
