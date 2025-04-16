from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class LinearCosts:
    def set_coefficients(self, coefficients: np.ndarray) -> None:
        self.coefficients = coefficients

    def set_offset(self, offset: float) -> None:
        self.offset = offset

    def get_coefficients(self) -> np.ndarray:
        return self.coefficients

    def get_offset(self) -> float:
        return self.offset
