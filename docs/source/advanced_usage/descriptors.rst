.. _tuning descriptors:

Tuning descriptors
==================

The data conversion shown in the basic MALA guide is straightforward from
an interface point of view, however it is non-trivial to determine the
correct hyperparameters for the bispectrum descriptors. Ideally, cutoff radius
and dimensionality of the descriptors should be chosen such that atomic
environments around each point in space are accurately represented.

Some physical intuition factors into this process, as detailed in
the MALA publication on `hyperparameter optimization <https://doi.org/10.1088/2632-2153/ac9956>`_.
There, it is discussed that too large cutoff radii lead to descriptors
that are too similar for different points in space, complicating the
ML training, while too small radii lead to descriptors not containing
enough physical information, and do not yield physically accurate models.
Likewise, the dimensionality of the expansion of the atomic density
needs to match the amount of information one wants to encode.

In the publication mentioned above, a method has been devised based on these
principles, the so called average cosine similarity distance (ACSD) analysis.
Based on cosine similarities between descriptor and target vectors for
distinct points, optimal descriptor hyperparameters can be determined; for
more detail, please refer to the publication.

Within MALA, this analysis is available as a special hyperparameter
optimization routine, as showcased in the example ``advanced/ex03

