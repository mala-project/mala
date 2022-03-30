"""Test whether the examples are still working."""

import runpy

import pytest


class TestExamples:
    @pytest.mark.examples
    def test_ex01(self):
        runpy.run_path("../examples/ex01_run_singleshot.py")

    @pytest.mark.examples
    def test_ex02(self):
        runpy.run_path("../examples/ex02_preprocess_data.py")

    @pytest.mark.examples
    def test_ex03(self):
        runpy.run_path("../examples/ex03_postprocess_data.py")

    @pytest.mark.examples
    def test_ex04(self):
        runpy.run_path("../examples/ex04_hyperparameter_optimization.py")

    @pytest.mark.examples
    def test_ex05(self):
        runpy.run_path("../examples/ex05_training_with_postprocessing.py")

    @pytest.mark.examples
    def test_ex06(self):
        runpy.run_path(
            "../examples/ex06_advanced_hyperparameter_optimization.py")

    @pytest.mark.examples
    def test_ex07(self):
        runpy.run_path("../examples/ex07_checkpoint_training.py")

    @pytest.mark.examples
    def test_ex08(self):
        runpy.run_path("../examples/ex08_checkpoint_hyperopt.py")

    @pytest.mark.examples
    def test_ex09(self):
        runpy.run_path("../examples/ex09_distributed_hyperopt.py")

    @pytest.mark.examples
    def test_ex10(self):
        runpy.run_path("../examples/ex10_tensor_board.py")

    @pytest.mark.examples
    def test_ex11(self):
        runpy.run_path("../examples/ex11_pass_single_feature.py")

    @pytest.mark.examples
    def test_ex12(self):
        runpy.run_path("../examples/ex12_run_predictions.py")

    @pytest.mark.examples
    def test_ex13(self):
        runpy.run_path("../examples/ex13_calculate_acsd.py")

    @pytest.mark.examples
    def test_ex14(self):
        runpy.run_path("../examples/ex14_advanced_networks.py")

    @pytest.mark.examples
    def test_ex15(self):
        runpy.run_path("../examples/ex15_ase_calculator.py")
