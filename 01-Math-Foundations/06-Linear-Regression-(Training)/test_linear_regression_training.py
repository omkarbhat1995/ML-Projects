import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

# Importing your solution assuming it's in a file named solution.py
# from solution import Solution 

class Solution:
    """Mocking the user's class directly in the test file for convenience."""
    def get_derivative(self, model_prediction, ground_truth, N, X, desired_weight):
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X, weights):
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(self, X, Y, num_iterations, initial_weights):
        for _ in range(num_iterations):
            predictions = self.get_model_prediction(X, initial_weights)
            for j in range(len(initial_weights)):
                gradient = self.get_derivative(predictions, Y, len(X), X, j)
                initial_weights[j] -= gradient * self.learning_rate
        return np.round(initial_weights, 5)

@pytest.fixture
def solver():
    return Solution()

# --- 15 Specific Test Cases ---

def test_zero_iterations(solver):
    X = np.array([[1.0]])
    Y = np.array([2.0])
    weights = np.array([0.0])
    result = solver.train_model(X, Y, 0, weights)
    assert_array_almost_equal(result, np.array([0.0]))

def test_perfect_initial_weights(solver):
    X = np.array([[1.0], [2.0]])
    Y = np.array([2.0, 4.0])
    weights = np.array([2.0])
    result = solver.train_model(X, Y, 10, weights)
    assert_array_almost_equal(result, np.array([2.0]))

def test_single_step_1d(solver):
    X = np.array([[1.0]])
    Y = np.array([2.0])
    weights = np.array([0.0])
    # Grad: -2*(2-0)*1/1 = -4. W = 0 - 0.01*(-4) = 0.04
    result = solver.train_model(X, Y, 1, weights)
    assert_array_almost_equal(result, np.array([0.04]))

def test_single_step_2d(solver):
    X = np.array([[1.0, 1.0]])
    Y = np.array([2.0])
    weights = np.array([0.0, 0.0])
    result = solver.train_model(X, Y, 1, weights)
    assert_array_almost_equal(result, np.array([0.04, 0.04]))

def test_zero_features(solver):
    X = np.array([[0.0, 0.0]])
    Y = np.array([5.0])
    weights = np.array([1.0, 2.0])
    result = solver.train_model(X, Y, 1, weights)
    assert_array_almost_equal(result, np.array([1.0, 2.0]))

def test_zero_targets(solver):
    X = np.array([[1.0]])
    Y = np.array([0.0])
    weights = np.array([1.0])
    # Grad: -2*(-1)*1 = 2. W = 1 - 0.01*2 = 0.98
    result = solver.train_model(X, Y, 1, weights)
    assert_array_almost_equal(result, np.array([0.98]))

def test_negative_weights(solver):
    X = np.array([[1.0]])
    Y = np.array([2.0])
    weights = np.array([-1.0])
    # Grad: -2*(3)*1 = -6. W = -1 - 0.01*(-6) = -0.94
    result = solver.train_model(X, Y, 1, weights)
    assert_array_almost_equal(result, np.array([-0.94]))

def test_negative_targets(solver):
    X = np.array([[1.0]])
    Y = np.array([-2.0])
    weights = np.array([0.0])
    # Grad: -2*(-2)*1 = 4. W = 0 - 0.01*4 = -0.04
    result = solver.train_model(X, Y, 1, weights)
    assert_array_almost_equal(result, np.array([-0.04]))

def test_multiple_samples_1d(solver):
    X = np.array([[1.0], [2.0]])
    Y = np.array([2.0, 4.0])
    weights = np.array([0.0])
    # Grad: -2/2 * (2*1 + 4*2) = -10. W = 0 - 0.01*(-10) = 0.1
    result = solver.train_model(X, Y, 1, weights)
    assert_array_almost_equal(result, np.array([0.1]))

def test_multiple_samples_2d(solver):
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    Y = np.array([2.0, 3.0])
    weights = np.array([0.0, 0.0])
    # Grad w0: -2/2 * (2*1) = -2 -> 0.02. Grad w1: -2/2 * (3*1) = -3 -> 0.03
    result = solver.train_model(X, Y, 1, weights)
    assert_array_almost_equal(result, np.array([0.02, 0.03]))

def test_bias_column(solver):
    X = np.array([[1.0, 1.0], [1.0, 2.0]])
    Y = np.array([3.0, 5.0])
    weights = np.array([0.0, 0.0])
    # Grad w0: -1*(3*1 + 5*1) = -8 -> 0.08. Grad w1: -1*(3*1 + 5*2) = -13 -> 0.13
    result = solver.train_model(X, Y, 1, weights)
    assert_array_almost_equal(result, np.array([0.08, 0.13]))

def test_identity_matrix(solver):
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    Y = np.array([1.0, 1.0])
    weights = np.array([1.0, 1.0])
    result = solver.train_model(X, Y, 10, weights)
    assert_array_almost_equal(result, np.array([1.0, 1.0]))

def test_decimal_features(solver):
    X = np.array([[0.5]])
    Y = np.array([1.0])
    weights = np.array([0.0])
    # Grad: -2/1 * (1 * 0.5) = -1. W = 0.01
    result = solver.train_model(X, Y, 1, weights)
    assert_array_almost_equal(result, np.array([0.01]))

def test_fractional_learning(solver):
    X = np.array([[2.0]])
    Y = np.array([1.0])
    weights = np.array([0.0])
    # Iter 1: Pred=0. Grad = -2*(1)*2 = -4 -> W=0.04
    # Iter 2: Pred=0.08. Grad = -2*(0.92)*2 = -3.68 -> W=0.04+0.0368 = 0.0768
    result = solver.train_model(X, Y, 2, weights)
    assert_array_almost_equal(result, np.array([0.0768]))

def test_convergence_to_known_value(solver):
    X = np.array([[1.0]])
    Y = np.array([10.0])
    weights = np.array([0.0])
    # With 500 iterations and 0.01 LR, it will essentially hit the target value
    result = solver.train_model(X, Y, 500, weights)
    assert_array_almost_equal(result, np.array([10.0]), decimal=4)
