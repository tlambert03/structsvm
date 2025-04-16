from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ilpy")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "uninstalled"


from .bundle_method import BundleMethod
from .soft_margin_loss import SoftMarginLoss
from .hamming_costs import HammingCosts

__all__ = ["BundleMethod", "SoftMarginLoss", "HammingCosts", "__version__"]
