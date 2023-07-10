"""Test whether the examples are still working."""

import runpy

import pytest


@pytest.mark.examples
class TestExamples:
    def test_ex01(self):
        runpy.run_path("../examples/basic/ex01_train_network.py")

    def test_ex02(self):
        runpy.run_path("../examples/basic/ex02_test_network.py")

    def test_ex03(self):
        runpy.run_path("../examples/basic/ex03_preprocess_data.py")

    def test_ex04(self):
        runpy.run_path("../examples/advanced/ex04_postprocess_data.py")

    def test_ex05(self):
        runpy.run_path("../examples/basic/ex04_hyperparameter_optimization.py")

    def test_ex06(self):
        runpy.run_path(
            "../examples/advanced/ex06_advanced_hyperparameter_optimization.py"
        )

    def test_ex07(self):
        runpy.run_path("../examples/advanced/ex01_checkpoint_training.py")

    def test_ex08(self):
        runpy.run_path("../examples/advanced/ex08_checkpoint_hyperopt.py")

    def test_ex09(self):
        runpy.run_path("../examples/advanced/ex09_distributed_hyperopt.py")

    def test_ex10(self):
        runpy.run_path("../examples/advanced/ex10_tensor_board.py")

    def test_ex11(self):
        runpy.run_path("../examples/advanced/ex11_pass_single_feature.py")

    def test_ex12(self):
        runpy.run_path("../examples/basic/ex05_run_predictions.py")

    def test_ex13(self):
        runpy.run_path("../examples/advanced/ex13_acsd.py")

    def test_ex14(self):
        runpy.run_path("../examples/advanced/ex14_advanced_networks.py")

    def test_ex15(self):
        runpy.run_path("../examples/basic/ex06_ase_calculator.py")

    def test_ex16(self):
        runpy.run_path("../examples/advanced/ex16_observables.py")

    def test_ex17(self):
        runpy.run_path(
            "../examples/advanced/ex17_visualize_electronic_structure.py")
