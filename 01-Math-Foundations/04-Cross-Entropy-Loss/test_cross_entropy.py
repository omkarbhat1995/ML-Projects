import unittest
import numpy as np
from numpy.typing import NDArray

class Solution:
    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        epsilon = 1e-7
        ypred_stable = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true*np.log(ypred_stable)+((1-y_true)*np.log(1-ypred_stable)))
        return round(loss, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        epsilon = 1e-7
        ypred_stable = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(np.sum(y_true*np.log(ypred_stable), axis=1))
        return round(loss, 4)

class TestCrossEntropyLoss(unittest.TestCase):
    def setUp(self):
        self.solution = Solution()

    # --- Binary Cross-Entropy (BCE) Tests ---

    def test_bce_standard(self):
        # Typical predictions with high confidence
        y_true = np.array([1, 0, 1], dtype=np.float64)
        y_pred = np.array([0.9, 0.1, 0.8], dtype=np.float64)
        self.assertEqual(self.solution.binary_cross_entropy(y_true, y_pred), 0.1446)

    def test_bce_perfect_prediction(self):
        # 100% accurate predictions. Loss should be essentially 0.
        # This also tests if np.clip prevents the log(0) NaN error.
        y_true = np.array([1, 0], dtype=np.float64)
        y_pred = np.array([1.0, 0.0], dtype=np.float64)
        self.assertEqual(self.solution.binary_cross_entropy(y_true, y_pred), 0.0)

    def test_bce_confidently_wrong(self):
        # Model predicts the exact opposite of the truth.
        # -log(1e-7) is approximately 16.11809
        y_true = np.array([1, 0], dtype=np.float64)
        y_pred = np.array([0.0, 1.0], dtype=np.float64)
        self.assertEqual(self.solution.binary_cross_entropy(y_true, y_pred), 16.1181)

    # --- Categorical Cross-Entropy (CCE) Tests ---

    def test_cce_standard(self):
        # Typical predictions across 3 classes for 2 samples
        y_true = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
        y_pred = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]], dtype=np.float64)
        self.assertEqual(self.solution.categorical_cross_entropy(y_true, y_pred), 0.1643)

    def test_cce_perfect_prediction(self):
        # 100% accurate predictions for one-hot encoded targets
        y_true = np.array([[1, 0, 0], [0, 0, 1]], dtype=np.float64)
        y_pred = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        self.assertEqual(self.solution.categorical_cross_entropy(y_true, y_pred), 0.0)

    def test_cce_confidently_wrong(self):
        # Model predicts the wrong class with 100% confidence
        y_true = np.array([[1, 0], [0, 1]], dtype=np.float64)
        y_pred = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        self.assertEqual(self.solution.categorical_cross_entropy(y_true, y_pred), 16.1181)

    def test_cce_uniform_distribution(self):
        # Model is completely unsure (predicts 33.33% for all 3 classes)
        # -log(0.3333333) is approximately 1.0986
        y_true = np.array([[1, 0, 0]], dtype=np.float64)
        y_pred = np.array([[0.3333333, 0.3333333, 0.3333333]], dtype=np.float64)
        self.assertEqual(self.solution.categorical_cross_entropy(y_true, y_pred), 1.0986)

if __name__ == '__main__':
    unittest.main()