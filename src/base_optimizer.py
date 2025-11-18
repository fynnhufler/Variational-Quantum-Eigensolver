import numpy as np
from abc import ABC, abstractmethod
class BaseOptimizer():
    def __init__(
        self,
        max_iter=100,
        random_state=None,
        store_history=True,
        initial_state=None,
        eps=1e-2
    ):
        self.max_iter = max_iter
        self.random_state = random_state
        self.min_diff = eps

        self.state = None if initial_state is None else np.array(initial_state, dtype=float)

        self.store_history = [] if store_history else None

    @abstractmethod
    def gradient_step(self, state: np.ndarray) -> np.ndarray:
        """
        Return the gradient (or descent direction) at the given state.
        """
        pass

    def step_size(self, state: np.ndarray, k: int) -> float:
        """Learning rate schedule. 
        Default: constant.
        """
        return 1.0

    def _step(self):
        """
        Calculate one iteration step and updates current state. 
        next_step = current_step - step_size * gradient_step
        """
        direction = self.gradient_step(self.state)
        eta = self.step_size(self.state, k)
        return self.state - eta * direction
    
    def _step(self, k: int) -> np.ndarray:
        direction = self.gradient_step(self.state)
        eta = self.step_size(self.state, k)
        return self.state - eta * direction

    def run(self, initial_state=None):
        """
        Run the optimization starting from `initial_state` if provided,
        otherwise from self.state.
        Returns the final state.
        """
        if initial_state is not None:
            self.state = np.array(initial_state, dtype=float)

        if self.state is None:
            raise ValueError("Initial state must be provided either in __init__ or run().")

        if self.store_history is not None:
            self.store_history.append(self.state.copy())

        for k in range(1, self.max_iter + 1):
            new_state = self._step(k)

            if self.store_history is not None:
                self.store_history.append(new_state.copy())

            # stopping criterion based on parameter change
            diff = np.linalg.norm(new_state - self.state)
            self.state = new_state

            if diff < self.min_diff:
                break

        return self.state
    

    #REFACTORING BEGINS HERE

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
        
        for iteration in range(self.max_iter):
            # Historie speichern
            self._store_iteration(params)
            
            # Gradient berechnen und Parameter updaten
            gradient = self.compute_gradient(params)
            new_params = self._update(params, gradient, iteration)
            
            # Konvergenzkriterium (optional)
            if np.abs(new_params - params) < self.eps:
                params = new_params
                break
            
            params = new_params
        
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
        """
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
        """


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
    
    def _apply_unitary(self, theta: float) -> np.ndarray:
        """U(θ) = exp(-iθY/2)"""
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        U = np.array([[c, s], [-s, c]], dtype=complex)
        return U @ self.initial_state
    
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
# KONKRETE VQE IMPLEMENTIERUNGEN (durch Mixins komponiert)
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