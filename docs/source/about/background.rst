Theoretical Background
======================

Density Functional Theory (DFT) is one of the most popular electronic structure
calculation methods due to its combination of reasonable accuracy and
computational cost. MALA works within the Born-Oppenheimer
approximation (i.e. fixed ionic positions :math:`\underline{\boldsymbol{R}}`)
and takes the electronic temperature
:math:`\tau` into account.

In DFT, the central quantity is the electronic density of a given system.
Within the Kohn-Sham framework, this density is given by

.. math::

    n(\boldsymbol{r}) = \sum_j f^\tau(\epsilon_j^\tau)\,
    |\phi_j(\boldsymbol{r})|^2 \; .

Here, :math:`\phi_j(\boldsymbol{r})` denote the Kohn-Sham wave functions,
which are given by the Kohn-Sham equations

.. math::

   \left[-\frac{1}{2}\nabla^2 +
    v_\mathrm{{\scriptscriptstyle S}}^\tau(\boldsymbol{r})\right]
    \phi_j(\boldsymbol{r}) = \epsilon_j^\tau \phi_j(\boldsymbol{r}) \; ,

which give a system of non-interacting particles restricted to reproduce
the density of the interacting system. The total free energy is evaluated using

.. math::

    A_\mathrm{total} =
    T_\mathrm{{\scriptscriptstyle S}}
    [\phi_j] -
    k_\mathrm{B}\tau S_\mathrm{{\scriptscriptstyle S}}
    [\epsilon_j]
    + E_\mathrm{{\scriptscriptstyle H}}
    [n] +
    E_\mathrm{{\scriptscriptstyle XC}}[n] +
    \int d\boldsymbol{r}\, n(\boldsymbol{r}) v(\boldsymbol{r})\; ,

with the external potential created by the ions :math:`v(\boldsymbol{r})`,
kinetic energy of the non-interacting system
:math:`T_\mathrm{{\scriptscriptstyle S}}`,
the electronic entropy of the non-interacting system
:math:`S_\mathrm{{\scriptscriptstyle S}}`,
the electrostatic interaction energy of the
density with itself :math:`E_\mathrm{{\scriptscriptstyle H}}`
and the exchange-correlation energy :math:`E_\mathrm{{\scriptscriptstyle XC}}`.

Forces and other quantities of interest can be derived from the total energy.

MALA aims to approximate this total energy by using Neural Networks (NN)
:math:`M` to learn the Local Density of States (LDOS), which is defined as

.. math::

    d(\epsilon, \boldsymbol{r}) = \sum_j |\phi_j(\boldsymbol{r})|^2 \delta(\epsilon-\epsilon_j^\tau) \; ,

and connected to the electronic density via

.. math::

    n(\boldsymbol{r}) =  \int d \epsilon\;  f^\tau(\epsilon) d(\epsilon, \boldsymbol{r})  \; ,

and the electronic density of states via

.. math::

    D(\epsilon) = \int d\boldsymbol{r} \; d(\epsilon, \boldsymbol{r})  \; .

These connections are important, since the
:math:`T_\mathrm{{\scriptscriptstyle S}}` and
:math:`S_\mathrm{{\scriptscriptstyle S}}` can be expressed in terms of
:math:`D`. MALA therefore evaluates

.. math::

    A_\mathrm{total} =
    E_b\big[D[d]\big] -
    k_\mathrm{B}\tau S_\mathrm{{\scriptscriptstyle S}}
    \big[D[d]\big]
    - E_\mathrm{{\scriptscriptstyle H}}
    \big[n[d]\big] +
    E_\mathrm{{\scriptscriptstyle XC}}\big[n[d]\big] +
    \int d\boldsymbol{r}\, n[d](\boldsymbol{r})
    v_\mathrm{{\scriptscriptstyle XC}}(\boldsymbol{r})\; ,

with the band energy :math:`E_b`.

If the LDOS can be learned appropriately accurate, MALA can therefore
evaluate energies analytically. To approximate the LDOS, an NN, :math:`M`, is
employed via

.. math::

    d(\epsilon, \boldsymbol{r}) = M(B(j, \boldsymbol{r}))
    [\boldsymbol{\lambda}] \; .

Thus, a network pass is performed *for every point in space*.
:math:`B(j, \boldsymbol{r})` are some descriptors that capture the local
environment. Often SNAP descriptors are used. :math:`\boldsymbol{\lambda}`
are the hyperparameters that characterize the neural network.

For more detailed information please refer to our
`recent publication <https://doi.org/10.1103/PhysRevB.104.035120>`_.
