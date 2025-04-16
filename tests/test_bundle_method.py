import ilpy
import numpy as np
import pytest

import structsvm as ssvm

try:
    ilpy.Solver(1, ilpy.VariableType.Continuous, preference=ilpy.Gurobi)
    HAVE_GUROBI = True
except RuntimeError:
    HAVE_GUROBI = False


@pytest.mark.skipif(not HAVE_GUROBI, reason="Currently only works on gurobi")
def test_quadratic() -> None:
    # f(x) = (x - 1)**2
    def value_gradient(x: np.ndarray) -> tuple[float, np.ndarray]:
        return (x[0] - 1.0) ** 2, 2 * (x - 1)

    bundle_method = ssvm.BundleMethod(
        value_gradient, dims=1, regularizer_weight=0.0001, eps=1e-5
    )

    w = bundle_method.optimize(max_iterations=100)
    assert round(w[0], 4) == 0.9990
