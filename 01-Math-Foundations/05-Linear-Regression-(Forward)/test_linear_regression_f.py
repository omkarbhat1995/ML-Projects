import pytest
import numpy as np
from numpy.typing import NDArray

# --- Your Solution Class ---
class Solution:
    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        prediction = np.matmul(X, weights)
        return np.round(prediction, 5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        mse = np.mean((model_prediction - ground_truth) ** 2)
        return np.round(mse, 5)

# --- Pytest Fixtures ---
@pytest.fixture
def solution():
    """Fixture to instantiate the Solution class for all tests."""
    return Solution()

# --- Test Cases for get_model_prediction (8 Tests) ---

def test_prediction_standard(solution):
    # Standard 2x2 matrix dot product
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    weights = np.array([0.5, 0.5])
    expected = np.array([1.5, 3.5])
    np.testing.assert_array_equal(solution.get_model_prediction(X, weights), expected)

def test_prediction_single_sample(solution):
    # A single row (1, m)
    X = np.array([[2.0, 3.0]])
    weights = np.array([1.0, 2.0])
    expected = np.array([8.0])
    np.testing.assert_array_equal(solution.get_model_prediction(X, weights), expected)

def test_prediction_zero_weights(solution):
    # Zero weights should result in zero predictions regardless of X
    X = np.array([[10.5, 20.2], [30.1, 40.8]])
    weights = np.array([0.0, 0.0])
    expected = np.array([0.0, 0.0])
    np.testing.assert_array_equal(solution.get_model_prediction(X, weights), expected)

def test_prediction_zero_X(solution):
    # Zero input features should result in zero predictions
    X = np.array([[0.0, 0.0], [0.0, 0.0]])
    weights = np.array([0.5, 0.9])
    expected = np.array([0.0, 0.0])
    np.testing.assert_array_equal(solution.get_model_prediction(X, weights), expected)

def test_prediction_negative_values(solution):
    # Testing negative values in both X and weights
    X = np.array([[-1.0, -2.0], [3.0, -4.0]])
    weights = np.array([-1.0, 1.0])
    expected = np.array([-1.0, -7.0])
    np.testing.assert_array_equal(solution.get_model_prediction(X, weights), expected)

def test_prediction_rounding(solution):
    # Ensuring the output is strictly rounded to 5 decimal places
    X = np.array([[1.111111, 2.222222]])
    weights = np.array([1.0, 1.0])
    # 1.111111 + 2.222222 = 3.333333 -> rounded to 5 places: 3.33333
    expected = np.array([3.33333])
    np.testing.assert_array_equal(solution.get_model_prediction(X, weights), expected)

def test_prediction_large_values(solution):
    # Testing large numbers to ensure no floating point overflow issues
    X = np.array([[1e5, 2e5]])
    weights = np.array([1e-5, 1e-5])
    expected = np.array([3.0])
    np.testing.assert_array_equal(solution.get_model_prediction(X, weights), expected)

def test_prediction_multiple_features(solution):
    # Testing a larger feature space (n=3, m=5)
    X = np.ones((3, 5))
    weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    # Sum of weights is 1.5, and X is all 1s
    expected = np.array([1.5, 1.5, 1.5])
    np.testing.assert_array_equal(solution.get_model_prediction(X, weights), expected)


# --- Test Cases for get_error (7 Tests) ---

def test_error_perfect_prediction(solution):
    # Zero error if predictions perfectly match ground truth
    preds = np.array([1.0, 2.5, 3.1])
    truth = np.array([1.0, 2.5, 3.1])
    assert solution.get_error(preds, truth) == 0.0

def test_error_standard(solution):
    # Standard Mean Squared Error calculation
    preds = np.array([2.0, 3.0])
    truth = np.array([1.0, 2.0])
    # MSE = ((1^2) + (1^2)) / 2 = 1.0
    assert solution.get_error(preds, truth) == 1.0

def test_error_negative_values(solution):
    # The squaring in MSE should handle negative differences correctly
    preds = np.array([1.0, 2.0])
    truth = np.array([-1.0, -2.0])
    # diffs = [2.0, 4.0] -> squares = [4.0, 16.0] -> mean = 10.0
    assert solution.get_error(preds, truth) == 10.0

def test_error_decimals(solution):
    # Calculating MSE with decimal differences
    preds = np.array([0.1, 0.9])
    truth = np.array([0.5, 0.5])
    # diffs = [-0.4, 0.4] -> squares = [0.16, 0.16] -> mean = 0.16
    assert solution.get_error(preds, truth) == 0.16

def test_error_rounding(solution):
    # Ensuring the MSE is strictly rounded to 5 decimal places
    preds = np.array([0.111111, 0.222222])
    truth = np.array([0.0, 0.0])
    # squares: [0.0123456, 0.0493826] -> mean: 0.0308641 -> rounded: 0.03086
    assert solution.get_error(preds, truth) == 0.03086

def test_error_single_element(solution):
    # Error calculation on a batch size of 1
    preds = np.array([3.0])
    truth = np.array([5.0])
    # MSE = (-2.0)^2 = 4.0
    assert solution.get_error(preds, truth) == 4.0

def test_error_large_differences(solution):
    # Ensuring stability with massive error distances
    preds = np.array([-1000.0])
    truth = np.array([1000.0])
    # diff = -2000 -> square = 4,000,000
    assert solution.get_error(preds, truth) == 4000000.0
