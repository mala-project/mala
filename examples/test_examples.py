"""Test whether the examples are still working."""

import runpy


class TestExamples:
    def test_ex01(self):
        runpy.run_path("ex01_run_singleshot.py")

    def test_ex02(self):
        runpy.run_path("ex02_preprocess_data.py")

    def test_ex03(self):
        runpy.run_path("ex03_postprocess_data.py")

    def test_ex04(self):
        runpy.run_path("ex04_hyperparameter_optimization.py")

    def test_ex05(self):
        runpy.run_path("ex05_training_with_postprocessing.py")

    def test_ex06(self):
        runpy.run_path("ex06_advanced_hyperparameter_optimization.py")

    def test_ex07(self):
        runpy.run_path("ex07_checkpoint_training.py")

    def test_ex08(self):
        runpy.run_path("ex08_checkpoint_hyperopt.py")

    def test_ex09(self):
        runpy.run_path("ex09_distributed_hyperopt.py")

    def test_ex10(self):
        runpy.run_path("ex10_tensor_board.py")

    def test_ex11(self):
        runpy.run_path("ex11_pass_single_feature.py")

    def test_ex12(self):
        runpy.run_path("ex12_run_predictions.py")

    def test_ex13(self):
        runpy.run_path("ex13_calculate_acsd.py")

    def test_ex14(self):
        runpy.run_path("ex14_advanced_networks.py")

    def test_ex15(self):
        runpy.run_path("ex15_ase_calculator.py")
