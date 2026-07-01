import unittest
import numpy as np
from numpy.typing import NDArray

class Solution:
    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # Subtract the max for numerical stability
        z_stable = z - np.max(z)
        # Compute the exponentials
        exp_z = np.exp(z_stable)
        # Normalize to get probabilities
        softmax_z = exp_z / np.sum(exp_z)
        return np.round(softmax_z, 4)

class TestSoftmax(unittest.TestCase):
    def setUp(self):
        self.solution = Solution()

    def test_standard_input(self):
        # Test with a standard array of logits
        z = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        expected = np.array([0.0900, 0.2447, 0.6652])
        np.testing.assert_array_almost_equal(self.solution.softmax(z), expected, decimal=4)

    def test_zero_input(self):
        # All zeros should result in a uniform distribution
        z = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        expected = np.array([0.3333, 0.3333, 0.3333])
        np.testing.assert_array_almost_equal(self.solution.softmax(z), expected, decimal=4)

    def test_numerical_stability(self):
        # Extremely large numbers that would cause overflow (np.exp(1000) is inf)
        # if max subtraction was not implemented.
        z = np.array([1000.0, 1001.0, 1002.0], dtype=np.float64)
        # Should output the exact same probabilities as [1.0, 2.0, 3.0]
        expected = np.array([0.0900, 0.2447, 0.6652])
        np.testing.assert_array_almost_equal(self.solution.softmax(z), expected, decimal=4)
        
    def test_negative_logits(self):
        # Test with entirely negative logits
        z = np.array([-3.0, -2.0, -1.0], dtype=np.float64)
        expected = np.array([0.0900, 0.2447, 0.6652])
        np.testing.assert_array_almost_equal(self.solution.softmax(z), expected, decimal=
