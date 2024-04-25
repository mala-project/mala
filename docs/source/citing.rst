Citing MALA
************

If you have you used MALA in your research, we kindly request you add proper
citations to your publications. To cite the MALA software itself, please
use the ``Cite this repository`` option on the GitHub MALA page.

Along with the code itself, please also cite the original MALA paper:


      .. code-block:: bibtex

            @article{MALA_foundational_paper,
                title = {Accelerating finite-temperature Kohn-Sham density functional theory with deep neural networks},
                author = {Ellis, J. A. and Fiedler, L. and Popoola, G. A. and Modine, N. A. and Stephens, J. A. and Thompson, A. P. and Cangi, A. and Rajamanickam, S.},
                journal = {Phys. Rev. B},
                volume = {104},
                issue = {3},
                pages = {035120},
                numpages = {17},
                year = {2021},
                month = {Jul},
                publisher = {American Physical Society},
                doi = {10.1103/PhysRevB.104.035120},
                url = {https://link.aps.org/doi/10.1103/PhysRevB.104.035120}
            }

If you perform a hyperparameter optimization during your research, please
cite the MALA hyperparameter paper:

      .. code-block:: bibtex

            @article{MALA_hyperparameters,
                doi = {10.1088/2632-2153/ac9956},
                url = {https://dx.doi.org/10.1088/2632-2153/ac9956},
                year = {2022},
                month = {oct},
                publisher = {IOP Publishing},
                volume = {3},
                number = {4},
                pages = {045008},
                author = {Lenz Fiedler and Nils Hoffmann and Parvez Mohammed and Gabriel A Popoola and Tamar Yovell and Vladyslav Oles and J Austin Ellis and Sivasankaran Rajamanickam and Attila Cangi},
                title = {Training-free hyperparameter optimization of neural networks for electronic structures in matter},
                journal = {Machine Learning: Science and Technology},
                abstract = {A myriad of phenomena in materials science and chemistry rely on quantum-level simulations of the electronic structure in matter. While moving to larger length and time scales has been a pressing issue for decades, such large-scale electronic structure calculations are still challenging despite modern software approaches and advances in high-performance computing. The silver lining in this regard is the use of machine learning to accelerate electronic structure calculations—this line of research has recently gained growing attention. The grand challenge therein is finding a suitable machine-learning model during a process called hyperparameter optimization. This, however, causes a massive computational overhead in addition to that of data generation. We accelerate the construction of neural network models by roughly two orders of magnitude by circumventing excessive training during the hyperparameter optimization phase. We demonstrate our workflow for Kohn–Sham density functional theory, the most popular computational method in materials science and chemistry.}
            }

Further, if you apply MALA at large length scales and/or across a temperature
range, please cite the respective transferability studies:

      .. code-block:: bibtex

            @article{MALA_sizetransfer,
                author={Fiedler, Lenz and Modine, Normand A. and Schmerler, Steve and Vogel, Dayton J. and Popoola, Gabriel A. and Thompson, Aidan P. and Rajamanickam, Sivasankaran and Cangi, Attila},
                title={Predicting electronic structures at any length scale with machine learning},
                journal={npj Computational Materials},
                year={2023},
                month={Jun},
                day={27},
                volume={9},
                number={1},
                pages={115},
                doi={10.1038/s41524-023-01070-z}}
            }

      .. code-block:: bibtex


            @article{MALA_temperaturetransfer,
                title = {Machine learning the electronic structure of matter across temperatures},
                author = {Fiedler, Lenz and Modine, Normand A. and Miller, Kyle D. and Cangi, Attila},
                journal = {Phys. Rev. B},
                volume = {108},
                issue = {12},
                pages = {125146},
                numpages = {16},
                year = {2023},
                month = {Sep},
                publisher = {American Physical Society},
                doi = {10.1103/PhysRevB.108.125146},
                url = {https://link.aps.org/doi/10.1103/PhysRevB.108.125146}
            }



