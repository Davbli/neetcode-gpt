import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        W1_T = np.transpose(W1)
        z1 = np.matmul(x, W1_T) + b1
        a1 = np.maximum(0, z1)
        z2 = np.matmul(a1, np.transpose(W2)) + b2
        n = len(y_true)
        dL_dyhat = (2/n) * (z2 - y_true)
        dyhat_dW2 = a1
        dyhat_da1 = W2
        da1_dz1 = (z1 > 0).astype(float)
        dz1_dW1 = x


        res = {}
        res["loss"] = round(np.mean((z2 - y_true)**2), 4)
        dL_dz2 = dL_dyhat                                    # (output_size,)
        dL_da1 = np.matmul(dL_dz2, dyhat_da1)               # (hidden_size,)
        dL_dz1 = dL_da1 * da1_dz1                            # (hidden_size,) elementwise
        res["dW1"] = np.round(np.outer(dL_dz1, dz1_dW1) + 0.0, 4).tolist()              # (hidden_size, input_size)
        res["db1"] = np.round(dL_dz1 + 0.0, 4).tolist()                                  # (hidden_size,)                                # (hidden_size,)
        res["dW2"] = np.round(np.outer(dL_dyhat, dyhat_dW2) + 0.0, 4).tolist()
        res["db2"] = np.round(dL_dyhat + 0.0, 4).tolist()

        return res