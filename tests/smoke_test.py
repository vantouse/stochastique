# tests/smoke_test.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from stochastique.models.FHN import model_FHN, equilibria_FHN, FitzHughNagumo
from stochastique.models.SN import model_SN, equilibria_SN, SaltzmanNicolis
from stochastique.core.solvers import TrajectorySolver
from stochastique.analysis.simulation import simulate_trajectory
from stochastique.core.numerical import find_jacobian, find_equilibrium, classify_equilibrium
from stochastique.core.dynamic import DynamicSystem2D
from stochastique.analysis.bifurcation import BifurcationAnalyzer
from stochastique.analysis.limit_cycles import LimitCycleAnalyzer
from stochastique.analysis.attraction import localize_basins_of_attraction


def test_fhn_model() -> None:
    """Test FitzHugh-Nagumo model evaluation and equilibrium finding."""
    print("Testing FHN model...")
    t = 0.0
    state = np.array([0.0, 0.0])
    deriv = model_FHN(t, state, delta=0.1, a=1.0)
    assert deriv.shape == (2,), f"Expected shape (2,), got {deriv.shape}"

    eqs = equilibria_FHN(delta=0.1, a=1.0)
    assert len(eqs) > 0, "Expected at least one equilibrium for FHN"
    
    # Test trajectory solver
    solver = TrajectorySolver(model_FHN)
    time_span = np.linspace(0, 10, 100)
    solution = solver.solve(state_init=eqs[0], time_span=time_span, params={"delta": 0.1, "a": 1.0})
    assert solution.shape == (100, 2), f"Expected solution shape (100, 2), got {solution.shape}"

    # Test class-based model interface
    fhn_model = FitzHughNagumo()
    assert fhn_model.default_params["delta"] == 0.1
    assert len(fhn_model.equilibria()) == 1
    print("FHN model test passed successfully.")


def test_sn_model() -> None:
    """Test Saltzman-Nicolis model evaluation and equilibria."""
    print("Testing SN model...")
    t = 0.0
    state = np.array([0.5, 0.5])
    deriv = model_SN(t, state, a=1.0, b=2.0)
    assert deriv.shape == (2,), f"Expected shape (2,), got {deriv.shape}"

    eqs = equilibria_SN(a=1.0, b=2.0)
    assert len(eqs) == 3, f"Expected 3 equilibria for SN, got {len(eqs)}"

    # Test stochastic simulation step
    rng = np.random.default_rng(42)
    trajectory = simulate_trajectory(
        model_func=model_SN,
        params={"a": 1.0, "b": 2.0},
        state_init=eqs[0],
        time_span=np.linspace(0, 5, 50),
        epsilon=0.01,
        noise_mask=np.array([1, 1]),
        rng=rng,
        method="rk4"
    )
    assert trajectory.shape == (50, 2), f"Expected trajectory shape (50, 2), got {trajectory.shape}"

    # Test class-based model interface
    sn_model = SaltzmanNicolis()
    assert sn_model.default_params["a"] == 1.0
    assert len(sn_model.equilibria()) == 3
    print("SN model test passed successfully.")


def test_numerical_routines() -> None:
    """Test numerical Jacobian and equilibrium classification."""
    print("Testing numerical routines...")
    params = {"a": 1.0, "b": 2.0}
    eqs = equilibria_SN(**params)
    eq = eqs[1] # (0, 0)
    
    J = find_jacobian(model_SN, 0.0, eq, params)
    assert J.shape == (2, 2), f"Expected Jacobian shape (2, 2), got {J.shape}"

    evals = np.linalg.eigvals(J)
    eq_type = classify_equilibrium(evals)
    assert isinstance(eq_type, str), "Equilibrium classification must return a string"
    print(f"Equilibrium (0,0) classified as: {eq_type}")
    print("Numerical routines test passed successfully.")


def test_dynamic_system() -> None:
    """Test DynamicSystem2D facade."""
    print("Testing DynamicSystem2D...")
    sys = DynamicSystem2D(model_FHN, params={"delta": 0.1, "a": 1.0})
    time_span = np.linspace(0, 5, 50)
    eqs = equilibria_FHN(delta=0.1, a=1.0)
    # Test DynamicSystem2D
    solution = sys.solve(eqs[0], time_span)
    assert solution.shape == (50, 2), f"Expected solution shape (50, 2), got {solution.shape}"
    print("DynamicSystem2D test passed successfully.")


def test_analysis_modules() -> None:
    """Test high-level analysis modules (Bifurcation, Limit Cycles, Basins)."""
    print("Testing analysis modules...")
    
    # 1. Bifurcation Analyzer
    bif = BifurcationAnalyzer(model_SN)
    fig, ax = plt.subplots()
    param_values = np.linspace(0.5, 1.5, 5)
    bif_points, eq_arr = bif.plot_bifurcation_diagram(
        param_name="a",
        param_values=param_values,
        state_init=lambda **kwargs: np.array([0.5, 0.5]),
        ax=ax,
        params={"a": 1.0, "b": 2.0}
    )
    plt.close(fig)
    assert isinstance(bif_points, np.ndarray)

    # 2. Limit Cycle Analyzer
    lc_analyzer = LimitCycleAnalyzer(model_FHN)
    time_span = np.linspace(0, 10, 100)
    sol = lc_analyzer.solve(np.array([1.0, 1.0]), time_span, {"delta": 0.1, "a": 1.0})
    crossings, ctimes = lc_analyzer.find_poincare_crossings(sol, time_span, x_section=0.0)
    assert isinstance(crossings, np.ndarray)

    # 3. Basins of Attraction
    sys = DynamicSystem2D(model_SN, params={"a": 1.0, "b": 2.0})
    eqs = equilibria_SN(a=1.0, b=2.0)
    fig, ax = plt.subplots()
    localize_basins_of_attraction(
        system=sys,
        equilibria=eqs[:2],
        time_span=np.linspace(0, 5, 20),
        ax=ax,
        bounds=(-2, 2),
        grid_size=5,
    )
    plt.close(fig)
    print("Analysis modules test passed successfully.")


if __name__ == "__main__":
    print("Starting smoke test suite...")
    test_fhn_model()
    test_sn_model()
    test_numerical_routines()
    test_dynamic_system()
    test_analysis_modules()
    print("All smoke tests passed successfully!")
