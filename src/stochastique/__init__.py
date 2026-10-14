# stochastique package
from .core import DynamicSystem2D, TrajectorySolver
from .models import FitzHughNagumo, SaltzmanNicolis, model_FHN, model_SN
from .analysis import BifurcationAnalyzer, LimitCycleAnalyzer, localize_basins_of_attraction, simulate_trajectory

__version__ = "0.0.1"
