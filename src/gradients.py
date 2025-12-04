# ============================================================================
# GRADIENT COMPUTATION MIXINS
# ============================================================================

from base_optimizer import *


class FiniteDifferenceGradient:
    """
    Mixin for numerical gradient computation via finite differences.
    """
    def __init__(self, *args, gradient_eps: float = 1e-6, **kwargs):
        self.gradient_eps = gradient_eps
        super().__init__(*args, **kwargs)
    
    def compute_gradient(self) -> np.ndarray:
        """Numerical gradient: (E(θ+ε) - E(θ)) / ε"""
        gradient = np.zeros(self.params_dim, dtype=float)
        energy_current = self.compute_expectation_value(self.state, self.hamilton)
        
        for i in range(self.params_dim):
            params_shifted = self.params.copy()
            params_shifted[i] += self.gradient_eps
            state_shifted = self.get_state(params_shifted)
            energy_shifted = self.compute_expectation_value(state_shifted, self.hamilton)
            gradient[i] = (energy_shifted - energy_current) / self.gradient_eps
        
        return gradient


class ParameterShiftGradient:
    """
    Exact gradient using parameter shift rule.
    For quantum circuits, this is exact (not an approximation).
    """
    def __init__(self, *args, shifts=None, **kwargs):
        self.shifts = shifts if shifts is not None else [np.pi/2]
        super().__init__(*args, **kwargs)
    
    def compute_gradient(self) -> np.ndarray:
        """Parameter shift: (E(θ+π/2) - E(θ-π/2)) / 2"""
        gradient = np.zeros(self.params_dim, dtype=float)
        
        for i in range(self.params_dim):
            params_plus = self.params.copy()
            params_plus[i] += np.pi/2
            state_plus = self.get_state(params_plus)
            
            params_minus = self.params.copy()
            params_minus[i] -= np.pi/2
            state_minus = self.get_state(params_minus)
            
            energy_plus = self.compute_expectation_value(state_plus, self.hamilton)
            energy_minus = self.compute_expectation_value(state_minus, self.hamilton)
            
            gradient[i] = (energy_plus - energy_minus) / 2.0
        
        return gradient


class SPSAGradient:
    """
    Simultaneous Perturbation Stochastic Approximation.
    Efficient for high-dimensional problems (only 2 function evaluations).
    """
    def __init__(self, *args, gradient_eps: float = 1e-6, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_eps = gradient_eps
    
    def compute_gradient(self) -> np.ndarray:
        """SPSA: (E(θ+ε·Δ) - E(θ-ε·Δ)) / (2·ε) · Δ"""
        # Random perturbation direction
        delta_k = 2 * np.random.randint(0, 2, size=self.params_dim) - 1
        
        params_plus = self.params + self.gradient_eps * delta_k
        state_plus = self.get_state(params_plus)
        
        params_minus = self.params - self.gradient_eps * delta_k
        state_minus = self.get_state(params_minus)
        
        energy_plus = self.compute_expectation_value(state_plus, self.hamilton)
        energy_minus = self.compute_expectation_value(state_minus, self.hamilton)
        
        gradient = (energy_plus - energy_minus) / (2 * self.gradient_eps) * delta_k
        return gradient
