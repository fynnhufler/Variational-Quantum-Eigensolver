import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector
from typing import Optional, List, Tuple, Any


# ============================================================================
# BASE OPTIMIZER
# ============================================================================

class BaseOptimizer(ABC):
    """
    Basis-Klasse für alle Optimierer.
    Definiert die allgemeine Struktur eines iterativen Optimierungsprozesses.
    """
    def __init__(
        self,
        max_iter: int = 100,
        eps: float = 1e-6,
        store_history: bool = True,
        random_state: Optional[int] = None
    ):
        self.max_iter = max_iter
        self.eps = eps
        self.random_state = random_state
        
        # History storage
        self.store_history_flag = store_history
        self.history_params: List[Any] = [] if store_history else None
        self.history_energy: List[float] = [] if store_history else None
        self.history_state: List[Any] = [] if store_history else None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    @abstractmethod
    def compute_gradient(self, params: Any) -> Any:
        """
        Berechnet den Gradienten bezüglich der Parameter.
        Muss in Subklassen implementiert werden.
        """
        pass
    
    @abstractmethod
    def compute_energy(self, params: Any) -> float:
        """
        Berechnet die Energie für gegebene Parameter.
        Muss in Subklassen implementiert werden.
        """
        pass
    
    @abstractmethod
    def get_state(self, params: Any) -> Any:
        """
        Gibt den Quantenzustand für gegebenes Theta / Parameter zurück.
        Muss in Subklassen implementiert werden.
        """
        pass
    
    def step_size(self, params: Any, iteration: int) -> float:
        """
        Learning rate schedule. Kann überschrieben werden.
        Default: konstant 0.01
        """
        return 0.01
    
    def _update(self, params: Any, gradient: Any, iteration: int) -> Any:
        """
        Standard Gradientenabstieg: θ_new = θ_old - η * ∇E
        """
        eta = self.step_size(params, iteration)
        return params - eta * gradient
    
    def _store_iteration(self, params: Any):
        """Speichert Historie falls aktiviert"""
        if self.store_history_flag:
            self.history_params.append(params)
            self.history_energy.append(self.compute_energy(params))
            self.history_state.append(self.get_state(params))
    
    def run(self, initial_params: Any) -> Tuple[Any, float]:
        """
        Führt die Optimierung aus.
        
        Returns:
            (optimale_parameter, optimale_energie)
        """
        params = initial_params
        energy_prev = self.compute_energy(params)
        
        for iteration in range(self.max_iter):
            # Historie speichern
            self._store_iteration(params)

            # Gradient berechnen und Parameter updaten
            gradient = self.compute_gradient(params)
            new_params = self._update(params, gradient, iteration)

            energy_new = self.compute_energy(new_params)
            
            # Konvergenzkriterium (optional)
            if np.abs(energy_new - energy_prev) < self.eps:
                params = new_params
                break
            
            #updates for next iteration
            params = new_params
            energy_prev = energy_new
        
        # Finale Werte speichern
        self._store_iteration(params)
        
        final_energy = self.compute_energy(params)
        return params, final_energy
    
    def plot_results(self):
        """Plottet die Optimierungshistorie"""
        if not self.store_history_flag:
            print("No history stored. Set store_history=True to enable plotting.")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot 1: Parameter und Energie
        ax1 = axes[0]
        ax1_twin = ax1.twinx()
        
        ax1.plot(self.history_params, 'g-', label='θ (params)', linewidth=2)
        ax1_twin.plot(self.history_energy, 'r-', label='Energy', linewidth=2)
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('θ', color='g')
        ax1_twin.set_ylabel('Energy', color='r')
        ax1.set_title('Parameter and Energy Evolution')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Zustandsentwicklung
        ax2 = axes[1]
        if self.history_state and hasattr(self.history_state[0], '__iter__'):
            states_array = np.array([s if isinstance(s, np.ndarray) else s.data 
                                   for s in self.history_state])
            
            if states_array.ndim > 1 and states_array.shape[1] >= 2:
                ax2.plot(np.abs(states_array[:, 0])**2, label='|⟨0|ψ⟩|²', linewidth=2)
                ax2.plot(np.abs(states_array[:, 1])**2, label='|⟨1|ψ⟩|²', linewidth=2)
                ax2.set_xlabel('Iteration')
                ax2.set_ylabel('Probability')
                ax2.set_title('State Evolution')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# ============================================================================
# GRADIENT COMPUTATION MIXINS
# ============================================================================

class FiniteDifferenceGradient:
    """
    Mixin für numerische Gradientenberechnung via finite Differenzen.
    """
    def __init__(self, *args, gradient_eps: float = 1e-6, **kwargs):
        self.gradient_eps = gradient_eps
        super().__init__(*args, **kwargs)
    
    def compute_gradient(self, params: Any) -> Any:
        """Numerischer Gradient: (E(θ+ε) - E(θ)) / ε"""
        energy_current = self.compute_energy(params)
        params_shifted = params + self.gradient_eps
        energy_shifted = self.compute_energy(params_shifted)
        return (energy_shifted - energy_current) / self.gradient_eps


# ============================================================================
# STEP SIZE STRATEGIES
# ============================================================================

class ConstantStepSize:
    """Konstante Lernrate"""
    def __init__(self, *args, learning_rate: float = 0.01, **kwargs):
        self.learning_rate = learning_rate
        super().__init__(*args, **kwargs)
    
    def step_size(self, params: Any, iteration: int) -> float:
        return self.learning_rate


class DecayingStepSize:
    """Abnehmende Lernrate: η_k = η_0 / (1 + decay * k)"""
    def __init__(self, *args, learning_rate: float = 0.1, decay: float = 0.01, **kwargs):
        self.learning_rate = learning_rate
        self.decay = decay
        super().__init__(*args, **kwargs)
    
    def step_size(self, params: Any, iteration: int) -> float:
        return self.learning_rate / (1.0 + self.decay * iteration)




# ============================================================================
# Quantum Natural Gradient Descent
# ============================================================================


# ============================================================================
# GENERALIZED PARAMETER SHIFT RULE
# ============================================================================

class ParameterShiftGradient:
    """Exact gradient using parameter shift rule"""
    def __init__(self, *args, shifts=None, **kwargs):
        self.shifts = shifts if shifts is not None else [np.pi/2]
        super().__init__(*args, **kwargs)  # Pass remaining args up

    def compute_gradient(self, params):
        params = np.atleast_1d(params)
        gradient = np.zeros_like(params)
        
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += np.pi/2
            
            params_minus = params.copy()
            params_minus[i] -= np.pi/2
            
            gradient[i] = (self.compute_energy(params_plus) - 
                          self.compute_energy(params_minus)) / 2.0
        
        return gradient[0] if len(gradient) == 1 else gradient


# ============================================================================
# GENERALIZED FISHER INFORMATION MATRIX
# ============================================================================
"""
class GeneralFisherInformationMixin:
    
    #Generalized Fisher Information Matrix for multi-parameter circuits.
    #
    #For n parameters, F is an n×n matrix:
    #F_ij(θ) = Re[⟨∂_i ψ|∂_j ψ⟩ - ⟨∂_i ψ|ψ⟩⟨ψ|∂_j ψ⟩]
    #
    #Computed via parameter shift rule.
    
    def __init__(self, *args, fisher_shifts: Optional[List[float]] = None,
                 fisher_reg: float = 1e-6, **kwargs):
        self.fisher_shifts = fisher_shifts if fisher_shifts is not None else [np.pi/2]
        self.fisher_reg = fisher_reg
        super().__init__(*args, **kwargs)
    
    def compute_fisher_information(self, params: np.ndarray) -> np.ndarray:
        
        #Compute the full Fisher Information Matrix.
        #
        #Args:
        #    params: Array of shape (n_params,)
        #    
        #Returns:
        #    F: Array of shape (n_params, n_params) - Fisher Information Matrix
        
        n_params = len(params) if hasattr(params, '__len__') else 1
        
        if n_params == 1:
            # Single parameter - return scalar (backward compatible)
            shift = self.fisher_shifts[0] if self.fisher_shifts else np.pi/2
            state_plus = self.get_state(params + shift)
            state_minus = self.get_state(params - shift)
            
            if hasattr(state_plus, 'data'):
                psi_plus = state_plus.data
                psi_minus = state_minus.data
            else:
                psi_plus = state_plus
                psi_minus = state_minus
            
            overlap = np.vdot(psi_plus, psi_minus)
            #common approximation
            F = 0.5 * (1.0 - np.abs(overlap)**2)
            
            return F + self.fisher_reg
        
        #multi-parameter case: compute full FIM
        F = np.zeros((n_params, n_params))
        
        #get states for shifted parameters
        states = {}
        states['center'] = self.get_state(params)
        
        for i in range(n_params):
            shift = self.fisher_shifts[i] if i < len(self.fisher_shifts) else np.pi/2
            
            params_plus = params.copy()
            params_plus[i] += shift
            states[f'plus_{i}'] = self.get_state(params_plus)
            
            params_minus = params.copy()
            params_minus[i] -= shift
            states[f'minus_{i}'] = self.get_state(params_minus)
        
        #extract numpy arrays (or qiskit)
        for key in states:
            if hasattr(states[key], 'data'):
                states[key] = states[key].data
        
        #compute FIM
        for i in range(n_params):
            for j in range(i, n_params):  # Use symmetry
                # Approximation using parameter shift
                # F_ij ≈ Re[⟨ψ_i+|ψ_j+⟩ - ⟨ψ_i+|ψ⟩⟨ψ|ψ_j+⟩]
                psi_i_plus = states[f'plus_{i}']
                psi_j_plus = states[f'plus_{j}']
                psi_center = states['center']
                
                # More accurate: 4-point formula
                psi_i_minus = states[f'minus_{i}']
                psi_j_minus = states[f'minus_{j}']
                
                #fubini-Study metric formula (via overlaps)
                #FIM can be written using this for pure states
                overlap_pp = np.vdot(psi_i_plus, psi_j_plus)
                overlap_pm = np.vdot(psi_i_plus, psi_j_minus)
                overlap_mp = np.vdot(psi_i_minus, psi_j_plus)
                overlap_mm = np.vdot(psi_i_minus, psi_j_minus)
                
                #central difference approximation
                F[i, j] = 0.125 * (
                    2.0 - np.real(overlap_pp) - np.real(overlap_mm)
                    - 1j * (np.imag(overlap_pm) - np.imag(overlap_mp))
                ).real
                
                #symmetric matrix
                if i != j:
                    F[j, i] = F[i, j]
        
        #regularization to diagonal
        F += self.fisher_reg * np.eye(n_params)
        
        return F
"""

class FisherInformationMixin:
    """Compute Fisher Information Matrix"""
    def __init__(self, *args, fisher_reg: float = 1e-6, **kwargs):
        self.fisher_reg = fisher_reg
        super().__init__(*args, **kwargs)

    def compute_fisher_information(self, params):
        params = np.atleast_1d(params)
        n = len(params)
        F = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                params_plus = params.copy()
                params_plus[i] += np.pi/2
                params_plus[j] += np.pi/2
                
                params_minus = params.copy()
                params_minus[i] -= np.pi/2
                params_minus[j] -= np.pi/2
                
                state_plus = self.get_state(params_plus)
                state_minus = self.get_state(params_minus)
                
                # Extract numpy arrays
                if hasattr(state_plus, 'data'):
                    state_plus = state_plus.data
                    state_minus = state_minus.data
                
                overlap = np.vdot(state_plus, state_minus)
                F[i, j] = 0.5 * (1.0 - np.abs(overlap)**2)
                F[j, i] = F[i, j]  # Symmetric
        
        # Add regularization
        F += self.fisher_reg * np.eye(n)
        
        return F[0, 0] if n == 1 else F


# ============================================================================
# GENERALIZED QUANTUM NATURAL GRADIENT
# ============================================================================

class QuantumNaturalGradient:
    """Update using natural gradient"""
    def _update(self, params, gradient, iteration):
        eta = self.step_size(params, iteration)
        F = self.compute_fisher_information(params)
        
        if np.isscalar(F):
            natural_gradient = gradient / F
        else:
            #we might have to use a different method to compute the inverse
            natural_gradient = np.linalg.solve(F, gradient)
        
        return params - eta * natural_gradient


# ============================================================================
# 2-param Numpy System
# ============================================================================

"""
class TwoQubitSystemNumpy(MultiParameterQuantumSystem):
    
    #Example: Two-parameter single-qubit system.
    #Ansatz: U(θ₁, θ₂) = Rz(θ₂) Ry(θ₁)
    #Hamiltonian: H = -X - Z
    
    def __init__(self, **kwargs):
        super().__init__(n_params=2, **kwargs)
    
    def _setup_hamiltonian(self):
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        self.H = -X - Z
    
    def _setup_initial_state(self):
        self.initial_state = np.array([1, 0], dtype=complex)
    
    def _build_circuit(self, params: np.ndarray) -> np.ndarray:
        
        #U(θ₁, θ₂) = Rz(θ₂) Ry(θ₁) |0⟩
        
        theta1, theta2 = params[0], params[1]
        
        # Ry(θ₁)
        c1 = np.cos(theta1 / 2)
        s1 = np.sin(theta1 / 2)
        Ry = np.array([[c1, -s1], [s1, c1]], dtype=complex)
        
        # Rz(θ₂)
        Rz = np.array([[np.exp(-1j * theta2 / 2), 0],
                       [0, np.exp(1j * theta2 / 2)]], dtype=complex)
        
        # Apply gates
        state = Ry @ self.initial_state
        state = Rz @ state
        
        return state
    
    def compute_energy(self, params: np.ndarray) -> float:
        state = self._build_circuit(params)
        return np.real(np.vdot(state, self.H @ state))
"""

# ============================================================================
# QUANTUM SYSTEMS
# ============================================================================

class OneQubitSystem(BaseOptimizer):
    """
    Abstrakte Basisklasse für Ein-Qubit-Systeme.
    Definiert gemeinsame Struktur für numpy und Qiskit Implementierungen.
    """
    def __init__(self, initial_theta: float = np.pi/2, **kwargs):
        super().__init__(**kwargs)
        self.theta = initial_theta
        self._setup_hamiltonian()
        self._setup_initial_state()
    
    @abstractmethod
    def _setup_hamiltonian(self):
        """Definiert den Hamiltonian H = -X - Z"""
        pass
    
    @abstractmethod
    def _setup_initial_state(self):
        """Definiert den initialen Zustand |0⟩"""
        pass
    
    @abstractmethod
    def _apply_unitary(self, theta: float) -> Any:
        """Wendet die parametrisierte Unitäre U(θ) auf den Zustand an"""
        pass
    
    def get_state(self, theta: float) -> Any:
        """Alias für _apply_unitary für konsistente API"""
        return self._apply_unitary(theta)
    
    @abstractmethod
    def compute_energy(self, theta: float) -> float:
        """Berechnet ⟨ψ(θ)|H|ψ(θ)⟩"""
        pass


class OneQubitSystemNumpy(OneQubitSystem):
    """
    Ein-Qubit VQE mit reinem NumPy.
    Hamiltonian: H = -X - Z
    Ansatz: U(θ) = exp(-iθY/2) = [[cos(θ/2), sin(θ/2)], [-sin(θ/2), cos(θ/2)]]
    """
    def _setup_hamiltonian(self):
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        self.H = -X - Z
    
    def _setup_initial_state(self):
        self.initial_state = np.array([1, 0], dtype=complex)
    """
    def _apply_unitary(self, theta: float) -> np.ndarray:
        #U(θ) = exp(-iθY/2)
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        U = np.array([[c, s], [-s, c]], dtype=complex)
        return U @ self.initial_state
    """
    def _apply_unitary(self, theta) -> np.ndarray:
        """
        U(θ) for single or multiple parameters
        - Single param: U(θ) = exp(-iθY/2) 
        - Two params: U(θ₁, θ₂) = Rz(θ₂) Ry(θ₁)
        """
        # Always convert to 1D array
        theta = np.atleast_1d(np.asarray(theta))
        
        if len(theta) == 1:
            # Single parameter: Ry rotation
            c = np.cos(theta[0] / 2)
            s = np.sin(theta[0] / 2)
            U = np.array([[c, s], [-s, c]], dtype=complex)
            return U @ self.initial_state
        
        elif len(theta) == 2:
            # Two parameters: Rz(θ₂) Ry(θ₁)
            c1 = np.cos(theta[0] / 2)
            s1 = np.sin(theta[0] / 2)
            Ry = np.array([[c1, -s1], [s1, c1]], dtype=complex)
            
            Rz = np.array([[np.exp(-1j * theta[1] / 2), 0],
                        [0, np.exp(1j * theta[1] / 2)]], dtype=complex)
            
            state = Ry @ self.initial_state
            state = Rz @ state
            return state
        
        else:
            raise ValueError(f"Expected 1 or 2 parameters, got {len(theta)}")
        
        
    def compute_energy(self, theta: float) -> float:
        """E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩"""
        state = self._apply_unitary(theta)
        return np.real(np.vdot(state, self.H @ state))


class OneQubitSystemQiskit(OneQubitSystem):
    """
    Ein-Qubit VQE mit Qiskit.
    Nutzt Qiskit's Statevector und Operator Klassen.
    """
    def _setup_hamiltonian(self):
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        self.H = Operator(-X - Z)
    
    def _setup_initial_state(self):
        qc = QuantumCircuit(1)
        self.initial_state = Statevector.from_instruction(qc)
    
    def _apply_unitary(self, theta: float) -> Statevector:
        """U(θ) = exp(-iθY/2)"""
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        U_matrix = np.array([[c, s], [-s, c]], dtype=complex)
        U = Operator(U_matrix)
        return self.initial_state.evolve(U, qargs=[0])
    
    def compute_energy(self, theta: float) -> float:
        """E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩"""
        state = self._apply_unitary(theta)
        return np.real(np.vdot(state.data, self.H.data @ state.data))


# ============================================================================
# KONKRETE VQE IMPLEMENTIERUNGEN
# ============================================================================

class VQE_Numpy_FiniteDiff_ConstStep(
    FiniteDifferenceGradient,
    ConstantStepSize,
    OneQubitSystemNumpy
):
    """VQE mit NumPy, finiten Differenzen und konstanter Schrittweite"""
    pass


class VQE_Qiskit_FiniteDiff_ConstStep(
    FiniteDifferenceGradient,
    ConstantStepSize,
    OneQubitSystemQiskit
):
    """VQE mit Qiskit, finiten Differenzen und konstanter Schrittweite"""
    pass


class VQE_Numpy_FiniteDiff_DecayStep(
    FiniteDifferenceGradient,
    DecayingStepSize,
    OneQubitSystemNumpy
):
    """VQE mit NumPy, finiten Differenzen und abnehmender Schrittweite"""
    pass

class QNGD_Numpy(
    ParameterShiftGradient,
    FisherInformationMixin,
    QuantumNaturalGradient,
    ConstantStepSize,
    OneQubitSystemNumpy
):
    """Quantum Natural Gradient Descent"""
    pass

"""

# ============================================================================
# Example
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("VQE Optimization - NumPy Implementation")
    print("=" * 60)
    
    # NumPy Variante
    vqe_numpy = VQE_Numpy_FiniteDiff_ConstStep(
        max_iter=500,
        learning_rate=0.01,
        gradient_eps=1e-6,
        store_history=True
    )
    
    optimal_theta_np, optimal_energy_np = vqe_numpy.run(initial_params=np.pi/2)
    
    print(f"\nOptimal θ: {optimal_theta_np:.6f}")
    print(f"Optimal Energy: {optimal_energy_np:.6f}")
    print(f"Theoretical minimum: {-np.sqrt(2):.6f}")
    
    vqe_numpy.plot_results()
    
    print("\n" + "=" * 60)
    print("VQE Optimization - Qiskit Implementation")
    print("=" * 60)
    
    # Qiskit Variante
    vqe_qiskit = VQE_Qiskit_FiniteDiff_ConstStep(
        max_iter=500,
        learning_rate=0.01,
        gradient_eps=1e-6,
        store_history=True
    )
    
    optimal_theta_qk, optimal_energy_qk = vqe_qiskit.run(initial_params=np.pi/2)
    
    print(f"\nOptimal θ: {optimal_theta_qk:.6f}")
    print(f"Optimal Energy: {optimal_energy_qk:.6f}")
    print(f"Theoretical minimum: {-np.sqrt(2):.6f}")
    
    vqe_qiskit.plot_results()

"""

if __name__ == "__main__":
    print("=" * 60)
    print("QNGD - Single Parameter")
    print("=" * 60)
    
    qngd = QNGD_Numpy(
        max_iter=100,
        learning_rate=0.1,
        fisher_reg=1e-6,
        store_history=True
    )
    
    theta_opt, energy_opt = qngd.run(initial_params=np.pi/2)
    print(f"Optimal θ: {theta_opt:.6f}")
    print(f"Optimal Energy: {energy_opt:.6f}")
    print(f"Theoretical: {-np.sqrt(2):.6f}")
    
    print("\n" + "=" * 60)
    print("QNGD - Two Parameters")
    print("=" * 60)
    
    qngd2 = QNGD_Numpy(
        max_iter=200,
        learning_rate=0.1,
        fisher_reg=1e-6,
        store_history=True
    )
    
    params_opt, energy_opt = qngd2.run(initial_params=np.array([np.pi/4, np.pi/3]))
    print(f"Optimal params: {params_opt}")
    print(f"Optimal Energy: {energy_opt:.6f}")
    print(f"Theoretical: {-np.sqrt(2):.6f}")
