import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector
from typing import Optional, List, Tuple, Any
from qiskit.circuit.library import RealAmplitudes, EfficientSU2

from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock


# ============================================================================
# BASE OPTIMIZER
# ============================================================================

class BaseOptimizer(ABC):
    """
    Base class for all optimizers
    Defines the general structure of an iterative optimization process
    """
    def __init__(
        self,
        max_iter: int = 100,
        eps_energy: float = 1e-6,
        store_history: bool = True,
        random_state: Optional[int] = None
    ):
        self.max_iter = max_iter
        self.eps_energy = eps_energy
        self.random_state = random_state
        
        # Current state tracking
        self.params = None
        self.state = None
        self.iteration = 0
        self.params_dim = None
        
        # History storage
        self.store_history_flag = store_history
        self.history_params: List[Any] = [] if store_history else None
        self.history_energy: List[float] = [] if store_history else None
        self.history_state: List[Any] = [] if store_history else None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    @abstractmethod
    def compute_gradient(self) -> Any:
        """
        Compute gradient with respect to parameters.
        Must be implemented in subclasses.
        """
        pass
    
    def compute_expectation_value(self, state: Statevector, operator: Operator) -> float:
        """E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩"""
        return np.real(np.vdot(state.data, operator.data @ state.data))
    
    def get_state(self, params: np.ndarray) -> Statevector:
        """Get quantum state for given parameters"""
        return self.initial_state.evolve(self.ansatz(params))
    
    def step_size(self) -> float:
        """
        Learning rate schedule. Can be overridden.
        Default: constant 0.01
        """
        return 0.01
    
    def _update(self, gradient: Any) -> Any:
        """
        Standard gradient descent: θ_new = θ_old - η * ∇E
        """
        eta = self.step_size()
        return self.params - eta * gradient
    
    def _store_iteration(self):
        """Store history if enabled"""
        if self.store_history_flag:
            self.history_params.append(self.params.copy())
            self.history_energy.append(self.compute_expectation_value(self.state, self.hamilton))
            self.history_state.append(self.state)
    
    def run(self, initial_params: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Execute the optimization.
        
        Returns:
            (optimal_parameters, optimal_energy)
        """
        self.params = np.atleast_1d(initial_params).copy()
        self.params_dim = len(self.params)
        
        for self.iteration in range(self.max_iter):
            if self.iteration % 10 == 0:
                print(f"Iteration {self.iteration}/{self.max_iter} ({100*self.iteration/self.max_iter:.1f}%)")
            
            # Get current state and store
            self.state = self.get_state(self.params)
            self._store_iteration()
            
            # Compute gradient and update
            gradient = self.compute_gradient()
            new_params = self._update(gradient)
            new_state = self.get_state(new_params)
            
            # Check convergence
            E_old = self.compute_expectation_value(self.state, self.hamilton)
            E_new = self.compute_expectation_value(new_state, self.hamilton)
            
            if np.abs(E_new - E_old) < self.eps_energy:
                print(f"Converged at iteration {self.iteration}")
                self.params = new_params
                self.state = new_state
                break
            
            self.params = new_params
        
        # Final state
        self.state = self.get_state(self.params)
        self._store_iteration()
        final_energy = self.compute_expectation_value(self.state, self.hamilton)
        
        return self.params, final_energy
    
    def plot_results(self):
        """Plot optimization history"""
        if not self.store_history_flag:
            print("No history stored. Set store_history=True to enable plotting.")
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        ax.plot(self.history_energy, 'b-', linewidth=2, label='Energy')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Energy', fontsize=12)
        ax.set_title('Energy Evolution During Optimization', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        plt.show()


# ============================================================================
# GRADIENT COMPUTATION MIXINS
# ============================================================================

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


# ============================================================================
# FISHER INFORMATION MATRIX
# ============================================================================

class FisherInformationMixin:
    """
    Compute Fisher Information Matrix using parameter shift rule.
    This is the quantum geometric tensor for pure states.
    """
    def __init__(self, *args, fisher_reg: float = 1e-6, **kwargs):
        self.fisher_reg = fisher_reg
        super().__init__(*args, **kwargs)
    
    def compute_fisher_information(self) -> np.ndarray:
        """
        Compute Fisher Information Matrix.
        F_ij = Re[⟨∂_i ψ|∂_j ψ⟩ - ⟨∂_i ψ|ψ⟩⟨ψ|∂_j ψ⟩]
        """
        n = self.params_dim
        F = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    # Diagonal: shift only parameter i
                    params_plus = self.params.copy()
                    params_plus[i] += np.pi/2
                    
                    params_minus = self.params.copy()
                    params_minus[i] -= np.pi/2
                    
                    state_plus = self.get_state(params_plus)
                    state_minus = self.get_state(params_minus)
                    
                    overlap = np.vdot(state_plus.data, state_minus.data)
                    F[i, i] = 0.5 * (1.0 - np.abs(overlap)**2)
                else:
                    # Off-diagonal: use 4-point formula
                    params_pp = self.params.copy()
                    params_pp[i] += np.pi/2
                    params_pp[j] += np.pi/2
                    
                    params_mm = self.params.copy()
                    params_mm[i] -= np.pi/2
                    params_mm[j] -= np.pi/2
                    
                    params_pm = self.params.copy()
                    params_pm[i] += np.pi/2
                    params_pm[j] -= np.pi/2
                    
                    params_mp = self.params.copy()
                    params_mp[i] -= np.pi/2
                    params_mp[j] += np.pi/2
                    
                    state_pp = self.get_state(params_pp)
                    state_mm = self.get_state(params_mm)
                    state_pm = self.get_state(params_pm)
                    state_mp = self.get_state(params_mp)
                    
                    overlap_pp_mm = np.vdot(state_pp.data, state_mm.data)
                    overlap_pm_mp = np.vdot(state_pm.data, state_mp.data)
                    
                    F[i, j] = 0.5 * (1.0 - np.real(overlap_pp_mm * np.conj(overlap_pm_mp)))
                    F[j, i] = F[i, j]  # Symmetric
        
        # Add regularization
        F += self.fisher_reg * np.eye(n)
        
        return F if n > 1 else F[0, 0]


# ============================================================================
# QUANTUM NATURAL GRADIENT
# ============================================================================

class QuantumNaturalGradient:
    """
    Update using natural gradient: θ_new = θ_old - η * F^(-1) * ∇E
    Includes numerical stability checks.
    """
    def __init__(self, *args, max_gradient_norm: float = 1.0, **kwargs):
        self.max_gradient_norm = max_gradient_norm
        super().__init__(*args, **kwargs)
    
    def _update(self, gradient: np.ndarray) -> np.ndarray:
        eta = self.step_size()
        F = self.compute_fisher_information()
        
        if np.isscalar(F):
            #Scalar case (single parameter)
            if F < 1e-10:
                natural_gradient = gradient
            else:
                natural_gradient = gradient / F
        else:
            #Matrix case - check condition number
            cond = np.linalg.cond(F)
            if cond > 1e12:
                #Too ill-conditioned, use regular gradient
                natural_gradient = gradient
            else:
                #Use pseudoinverse for stability
                try:
                    natural_gradient = np.linalg.pinv(F, rcond=1e-10) @ gradient
                except:
                    natural_gradient = gradient
        
        #Clip gradient norm
        if np.isscalar(natural_gradient):
            grad_norm = abs(natural_gradient)
            if grad_norm > self.max_gradient_norm:
                natural_gradient = self.max_gradient_norm * np.sign(natural_gradient)
        else:
            grad_norm = np.linalg.norm(natural_gradient)
            if grad_norm > self.max_gradient_norm:
                natural_gradient = natural_gradient * (self.max_gradient_norm / grad_norm)
        
        return self.params - eta * natural_gradient


# ============================================================================
# STEP SIZE STRATEGIES
# ============================================================================

class ConstantStepSize:
    """Constant learning rate"""
    def __init__(self, *args, learning_rate: float = 0.01, **kwargs):
        self.learning_rate = learning_rate
        super().__init__(*args, **kwargs)
    
    def step_size(self) -> float:
        return self.learning_rate


class DecayingStepSize:
    """Decaying learning rate: η_k = η_0 / (1 + decay * k)"""
    def __init__(self, *args, learning_rate: float = 0.1, decay: float = 0.01, **kwargs):
        self.learning_rate = learning_rate
        self.decay = decay
        super().__init__(*args, **kwargs)
    
    def step_size(self) -> float:
        return self.learning_rate / (1.0 + self.decay * self.iteration)


class Adam:
    """Adam optimizer with adaptive learning rates"""
    def __init__(self, *args, learning_rate: float = 0.01, beta1: float = 0.9, 
                 beta2: float = 0.999, eps: float = 1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.eta = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0  #Adam iteration counter
    
    def _update(self, gradient: np.ndarray) -> np.ndarray:
        #Lazy initialization
        if self.m is None or self.v is None:
            self.m = np.zeros_like(self.params, dtype=float)
            self.v = np.zeros_like(self.params, dtype=float)
        
        self.t += 1
        
        #Update biased first and second moment estimates
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (gradient * gradient)
        
        #Bias correction
        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)
        
        #Parameter update
        params_new = self.params - self.eta * m_hat / (np.sqrt(v_hat) + self.eps)
        return params_new


# ============================================================================
# ANSATZ CLASSES
# ============================================================================

class TrivialAnsatz:
    """Simple single-qubit rotations: U = ⊗_i Ry(θ_i)"""
    def ansatz(self, theta: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.dim)
        for i in range(self.dim):
            qc.ry(theta[i], i)
        return qc


class RealAmplitudesAnsatz:
    """Hardware-efficient ansatz with real amplitudes"""
    def __init__(self, *args, reps: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.reps = reps
    
    def ansatz(self, theta: np.ndarray) -> QuantumCircuit:
        ansatz = RealAmplitudes(
            num_qubits=self.dim,
            reps=self.reps,
            entanglement='full',
            insert_barriers=True
        ).decompose()
        ansatz_bound = ansatz.assign_parameters(theta)
        return ansatz_bound


class ComplexAnsatz:
    """Hardware-efficient ansatz with complex amplitudes (EfficientSU2)"""
    def __init__(self, *args, reps: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.reps = reps
    
    def ansatz(self, theta: np.ndarray) -> QuantumCircuit:
        ansatz = EfficientSU2(
            num_qubits=self.dim,
            reps=self.reps,
            entanglement='full',
            insert_barriers=True
        ).decompose()
        ansatz_bound = ansatz.assign_parameters(theta)
        return ansatz_bound


# ============================================================================
# QUANTUM SYSTEMS
# ============================================================================

class OneQubitSystem(BaseOptimizer):
    """Single qubit system with H = -X - Z"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dim = 1
        self._setup_hamiltonian()
        self._setup_initial_state()
    
    def _setup_hamiltonian(self):
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        self.hamilton = Operator(-X - Z)
    
    def _setup_initial_state(self):
        qc = QuantumCircuit(1)
        self.initial_state = Statevector.from_instruction(qc)


class IsingModel(BaseOptimizer):
    """
    n-dimensional Ising model.
    H = -Σ_i h_i Z_i - Σ_{i<j} J_{ij} Z_i Z_j
    """
    def __init__(self, J: np.ndarray, h: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self.dim = J.shape[0]
        self._setup_hamiltonian(J, h)
        self._setup_initial_state()
    
    def _setup_hamiltonian(self, J: np.ndarray, h: np.ndarray):
        hamilton = np.zeros((2**self.dim, 2**self.dim), dtype=complex)
        
        for i in range(self.dim):
            hamilton = hamilton - h[i] * self._pauli_product(i, i)
            for j in range(i+1, self.dim):
                hamilton = hamilton - J[i, j] * self._pauli_product(i, j)
        
        self.hamilton = Operator(hamilton)
    
    def _setup_initial_state(self):
        self.initial_state = Statevector.from_instruction(QuantumCircuit(self.dim))
    
    def _pauli_product(self, i: int, j: int) -> np.ndarray:
        """Compute tensor product with Z at positions i and j"""
        Id = np.array([[1, 0], [0, 1]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        paulis = [Id for _ in range(self.dim)]
        paulis[i] = Z
        paulis[j] = Z
        
        result = np.array([[1]], dtype=complex)
        for pauli in paulis:
            result = np.kron(result, pauli)
        
        return result


class HydrogenMolecule(BaseOptimizer):
    """
    Hydrogen molecule (H2) using PySCF and Jordan-Wigner mapping.
    """
    def __init__(self, distance: float, **kwargs):
        super().__init__(**kwargs)
        self.dim = 4  #4 qubits for H2 with sto-3g
        self.distance = distance
        self._setup_hamiltonian(distance)
        self._setup_initial_state()
    
    def _setup_hamiltonian(self, distance: float):
        #Run PySCF
        driver = PySCFDriver(
            atom=f"H 0 0 {-distance/2}; H 0 0 {distance/2}",
            basis="sto3g",
            charge=0,
            spin=0,
            unit=DistanceUnit.ANGSTROM,
        )
        problem = driver.run()
        
        #Get fermionic Hamiltonian
        second_q_op = problem.hamiltonian.second_q_op()
        
        #Map to qubits using Jordan-Wigner
        self.mapper = JordanWignerMapper()
        qubit_op = self.mapper.map(second_q_op)
        
        #Convert to matrix and add nuclear repulsion
        H_el = qubit_op.to_matrix()
        enuc = problem.nuclear_repulsion_energy
        self.hamilton = Operator(H_el + enuc * np.eye(H_el.shape[0], dtype=complex))
    
    def _setup_initial_state(self):
        #Use Hartree-Fock initial state
        hf_circuit = HartreeFock(
            num_spatial_orbitals=2,
            num_particles=(1, 1),
            qubit_mapper=self.mapper,
        )
        self.initial_state = Statevector.from_instruction(hf_circuit)


# ============================================================================
# VQE IMPLEMENTATIONS
# ============================================================================

#SIngle Qubit Systems

class VQE_OneQubit_FiniteDiff_Const(
    FiniteDifferenceGradient,
    ConstantStepSize,
    TrivialAnsatz,
    OneQubitSystem
):
    """Single qubit VQE with finite differences and constant step size"""
    pass


class VQE_OneQubit_PSR_Adam(
    ParameterShiftGradient,
    Adam,
    RealAmplitudesAnsatz,
    OneQubitSystem
):
    """Single qubit VQE with parameter shift rule and Adam optimizer"""
    pass


class VQE_OneQubit_QNG(
    ParameterShiftGradient,
    FisherInformationMixin,
    QuantumNaturalGradient,
    DecayingStepSize,
    RealAmplitudesAnsatz,
    OneQubitSystem
):
    """Single qubit VQE with Quantum Natural Gradient Descent"""
    pass


# --- Ising Model ---

class VQE_Ising_PSR_Adam(
    ParameterShiftGradient,
    Adam,
    RealAmplitudesAnsatz,
    IsingModel
):
    """Ising model VQE with parameter shift rule and Adam"""
    pass


class VQE_Ising_QNG(
    ParameterShiftGradient,
    FisherInformationMixin,
    QuantumNaturalGradient,
    DecayingStepSize,
    RealAmplitudesAnsatz,
    IsingModel
):
    """Ising model VQE with Quantum Natural Gradient Descent"""
    pass


class VQE_Ising_SPSA_Adam(
    SPSAGradient,
    Adam,
    RealAmplitudesAnsatz,
    IsingModel
):
    """Ising model VQE with SPSA (efficient for high dimensions) and Adam"""
    pass


#H-Molecule

class VQE_H2_QNG(
    ParameterShiftGradient,
    FisherInformationMixin,
    QuantumNaturalGradient,
    DecayingStepSize,
    RealAmplitudesAnsatz,
    HydrogenMolecule
):
    """H2 molecule VQE with Quantum Natural Gradient Descent"""
    pass


class VQE_H2_PSR_Adam(
    ParameterShiftGradient,
    Adam,
    RealAmplitudesAnsatz,
    HydrogenMolecule
):
    """H2 molecule VQE with parameter shift rule and Adam"""
    pass


# ============================================================================
# EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EXAMPLE 1: Single Qubit System with QNG")
    print("=" * 70)
    
    vqe = VQE_OneQubit_QNG(
        max_iter=100,
        learning_rate=0.05,
        decay=0.01,
        fisher_reg=1e-6,
        max_gradient_norm=1.0,
        store_history=True,
        reps=0
    )
    
    theta_opt, E_opt = vqe.run(initial_params=np.array([np.pi/2]))
    
    print(f"\nOptimal θ: {theta_opt}")
    print(f"Optimal Energy: {E_opt:.6f}")
    print(f"Theoretical minimum: {-np.sqrt(2):.6f}")
    print(f"Error: {abs(E_opt + np.sqrt(2)):.6e}")
    
    vqe.plot_results()
    
    
    print("\n" + "=" * 70)
    print("EXAMPLE 2: 3-Qubit Ising Model with Adam")
    print("=" * 70)
    
    J = np.array([
        [0.0,  1.0,  0.0],
        [1.0,  0.0,  0.8],
        [0.0,  0.8,  0.0],
    ], dtype=float)
    
    h = np.array([0.2, -0.1, 0.05], dtype=float)
    
    vqe_ising = VQE_Ising_PSR_Adam(
        max_iter=200,
        learning_rate=0.05,
        store_history=True,
        reps=1,
        J=J,
        h=h
    )
    
    #Initialize parameters
    n_params = (vqe_ising.reps + 1) * 3  # 3 qubits
    theta_init = np.random.uniform(0, 2*np.pi, n_params)
    
    theta_opt, E_opt = vqe_ising.run(initial_params=theta_init)
    
    print(f"\nOptimal θ: {theta_opt}")
    print(f"Optimal Energy: {E_opt:.6f}")
    
    vqe_ising.plot_results()
    
    
    print("\n" + "=" * 70)
    print("EXAMPLE 3: H2 Potential Energy Curve with QNG")
    print("=" * 70)
    
    distances = np.linspace(0.5, 2.5, 10)
    energies = []
    
    for d in distances:
        print(f"\nDistance: {d:.2f} Å")
        
        vqe_h2 = VQE_H2_PSR_Adam(
            max_iter=100,
            learning_rate=0.05,
            store_history=False,
            reps=1,
            distance=d
        )
        
        n_params = (vqe_h2.reps + 1) * 4  # 4 qubits
        theta_init = np.random.uniform(0, 2*np.pi, n_params)
        
        _, E_opt = vqe_h2.run(initial_params=theta_init)
        energies.append(E_opt)
        
        print(f"Energy: {E_opt:.6f} Ha")
    
    #Plot potential energy curve
    plt.figure(figsize=(10, 6))
    plt.plot(distances, energies, 'o-', linewidth=2, markersize=8)
    plt.xlabel('H-H Distance (Å)', fontsize=12)
    plt.ylabel('Energy (Hartree)', fontsize=12)
    plt.title('H₂ Potential Energy Curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)