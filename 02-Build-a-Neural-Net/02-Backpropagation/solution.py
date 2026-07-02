import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        z = np.dot(x,w)+b
        y_hat = self.sigmoid(z)
        #L = 0.5 * (y_hat-y_true)**2
        error = (y_hat-y_true)
        sigmoid_derivation = y_hat*(1-y_hat)
        delta = error *sigmoid_derivation

        dl_dw = np.round(delta * x,5)
        dl_db = np.round(delta,5)
        return (dl_dw, dl_db)