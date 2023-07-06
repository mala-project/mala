Tuning the hyperparameters
==========================

MALA provides three ways to tune the hyperparaneters of a model.
The basic usage is the same:

      .. code-block:: python

          import mala

          parameters = mala.Parameters()
          datahandler = mala.DataHandler(parameters)
          datahandler.add_snapshot(...)
          hyperp_optimizer = mala.HyperOpt(parameters, datahandler)
          hyperp_optimizer.add_hyperparameter(...)
          hyperp_optimizer.perform_study()
          hyperp_optimizer.set_optimal_parameters()


COMING SOON: We are currently in the process of publishing a research article
on hyperparameter optimization in MALA. The preprint can be accessed at
https://arxiv.org/abs/2202.09186.

Optuna
*******

Optuna is a powerful library for hyperparameter optimization. It uses as
a sampler that progressively learns from proposed networks to eventually
identify the optimal one.
To use it, set ``parameters.hyperparameters.hyper_opt_method="optuna"``
Most Optuna customizations are accesible through MALA, including the support
for SQL based distributed hyperparameter studies. See ``ex04_hyperparameter_optimization``
and ``ex09_distributed_hyperopt``.

NASWOT
******

Neural Architecture Search Without Training (NASWOT) is a
training-free hyperparameter optimization.
To use it, set ``parameters.hyperparameters.hyper_opt_method="naswot"``
As no network training is required, this method is fast compared to Optuna
and OAT. On the downside, only architecture related hyperparameters will
be optimized. See also ``ex06_advanced_hyperparameter_optimization``.

OAT
***

Orthogonal Array Tuning (OAT) provides a middleground between Optuna and
NASWOT.
To use it, set ``parameters.hyperparameters.hyper_opt_method="oat"``
It aims to sample the space of available hyperparameters using an
*orthogonal array* and performs a range analysis afterwards.
See also ``ex06_advanced_hyperparameter_optimization``.
