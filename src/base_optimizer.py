import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector
from typing import Optional, List, Tuple, Any
from qiskit.circuit.library import RealAmplitudes
from qiskit.circuit.library import EfficientSU2


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

            E_old, E_new = self.compute_energy(params), self.compute_energy(new_params)
            
            # Konvergenzkriterium (optional)
            if np.abs(E_new - E_old) < self.eps:
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
        plt.show() 
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
        gradient=np.zeros(params.size, dtype=float)
        for i in range(params.size):
            energy_current = self.compute_energy(params)
            params_shifted=params.copy()
            params_shifted[i]=params[i]+self.gradient_eps
            energy_shifted = self.compute_energy(params_shifted)
            gradient[i]=(energy_shifted-energy_current)/self.gradient_eps
        return gradient
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


class adam:
    """Adam optimizer"""
    pass


# ============================================================================
# QUANTUM SYSTEMS
# ============================================================================

class OneQubitSystem(BaseOptimizer):
    """
    Abstrakte Basisklasse für Ein-Qubit-Systeme.
    Definiert gemeinsame Struktur für numpy und Qiskit Implementierungen.
    """
    def __init__(self, initial_theta: np.ndarray = np.array([np.pi/2]), **kwargs):
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
    
#    @abstractmethod
#    def _apply_unitary(self, theta) -> Any:
#        """Wendet die parametrisierte Unitäre U(θ) auf den Zustand an"""
#        pass
    
    @abstractmethod
    def get_state(self, theta) -> Any:
        pass
    
    @abstractmethod
    def compute_energy(self, theta) -> float:
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
    
    def get_state(self, theta) -> np.ndarray:
        """U(θ) = exp(-iθY/2)"""
        c = np.cos(theta[0] / 2)
        s = np.sin(theta[0] / 2)
        U = np.array([[c, s], [-s, c]], dtype=complex)
        return U @ self.initial_state
    
    def compute_energy(self, theta) -> float:
        """E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩"""
        state = self.get_state(theta)
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
    
    def get_state(self, theta) -> Statevector:
        """U(θ) = exp(-iθY/2)"""
        c = np.cos(theta[0] / 2)
        s = np.sin(theta[0] / 2)
        U_matrix = np.array([[c, s], [-s, c]], dtype=complex)
        U = Operator(U_matrix)
        return self.initial_state.evolve(U, qargs=[0])
    
    def compute_energy(self, theta) -> float:
        """E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩"""
        state = self.get_state(theta)
        return np.real(np.vdot(state.data, self.H.data @ state.data))
    

class n_dim_ising(BaseOptimizer):
    def __init__(self,J,h,**kwargs):
        super().__init__(**kwargs)
        self.dim=J.shape[0]
        self.Hamilton=self.Ham(J,h)
        self.Hamilton=Operator(self.Hamilton)
        self.state=Statevector.from_instruction(QuantumCircuit(self.dim))

    def Ham(self,J,h):
        Hamilton=np.zeros((2**self.dim,2**self.dim))
        for i in range(self.dim):
            Hamilton=Hamilton-h[i]*self.pauli_i_j(i,i)
            for j in range(i+1,self.dim):
                Hamilton=Hamilton-J[i,j]*self.pauli_i_j(i,j)
        return Hamilton


    def pauli_i_j(self,i,j):
        Id = np.array([[1,0],[0,1]])
        Z=np.array([[1,0],[0,-1]])
        tens_pauli=np.array([[1]])
        pauli_z_i_j = [Id for _ in range(self.dim)]
        pauli_z_i_j[i]=Z
        pauli_z_i_j[j]=Z
        for i in range(self.dim):
            tens_pauli=np.kron(tens_pauli,pauli_z_i_j[i])
        return tens_pauli

    @abstractmethod
    def unitary(self,theta):
        pass
#        u=np.array([[1]])
#        for i in range(self.dim):
#            u=np.kron(u,np.array([[np.cos(theta[i]/2), -np.sin(theta[i]/2)],[np.sin(theta[i]/2),  np.cos(theta[i]/2)]]))
#        return Operator(u)
    
    def get_state(self,theta):
        u=self.unitary(theta)
        return self.state.evolve(u)
    
    def compute_energy(self,theta)-> float:
        state = self.get_state(theta)
        energy=np.vdot(state.data,self.Hamilton.data @ state.data)
        return np.real(energy)
    

class n_dim_ising_triv_ansatz(n_dim_ising):
    def unitary(self,theta):
        u=np.array([[1]])
        for i in range(self.dim):
            u=np.kron(u,np.array([[np.cos(theta[i]/2), -np.sin(theta[i]/2)],[np.sin(theta[i]/2),  np.cos(theta[i]/2)]]))
        return Operator(u)



class n_dim_ising_real_ansatz(n_dim_ising):
    def __init__(self,reps,**kwargs):
        super().__init__(**kwargs)
        self.reps=reps

    def unitary(self,theta):
        num_qubits = self.dim
        reps = self.reps
        entanglement = 'full'

        ansatz = RealAmplitudes(num_qubits=num_qubits,reps=reps,entanglement=entanglement,insert_barriers=True,).decompose()
        ansatz_bound = ansatz.assign_parameters(theta)
        return ansatz_bound
    

class n_dim_ising_complex_ansatz(n_dim_ising):
    def __init__(self,reps,**kwargs):
        super().__init__(**kwargs)
        self.reps=reps

    def unitary(self,theta):
        num_qubits = self.dim
        reps = self.reps
        entanglement = 'full'

        ansatz = EfficientSU2(num_qubits=num_qubits,reps=reps,entanglement=entanglement,insert_barriers=True).decompose()
        ansatz_bound = ansatz.assign_parameters(theta)
        return ansatz_bound

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


class VQE_Ising_triv_FiniteDiff_ConstStep(
    FiniteDifferenceGradient,
    ConstantStepSize,
    n_dim_ising_triv_ansatz
):
    """VQE mit NumPy, finiten Differenzen und konstanter Schrittweite"""
    pass



class VQE_Ising_real_FiniteDiff_ConstStep(
    FiniteDifferenceGradient,
    ConstantStepSize,
    n_dim_ising_real_ansatz
):
    """VQE mit NumPy, finiten Differenzen und konstanter Schrittweite"""
    pass


class VQE_Ising_complex_FiniteDiff_ConstStep(
    FiniteDifferenceGradient,
    ConstantStepSize,
    n_dim_ising_complex_ansatz
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


#class VQE_Numpy_FiniteDiff_DecayStep(
#    FiniteDifferenceGradient,
#    DecayingStepSize,
#    OneQubitSystemNumpy
#):
#    """VQE mit NumPy, finiten Differenzen und abnehmender Schrittweite"""
#    pass





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
    
    optimal_theta_np, optimal_energy_np = vqe_numpy.run(initial_params=np.array([np.pi/2]))
    
    print(f"\nOptimal θ: {optimal_theta_np}")
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
    
    optimal_theta_qk, optimal_energy_qk = vqe_qiskit.run(initial_params=np.array([np.pi/2]))
    
    print(f"\nOptimal θ: {optimal_theta_np}")
    print(f"Optimal Energy: {optimal_energy_qk:.6f}")
    print(f"Theoretical minimum: {-np.sqrt(2):.6f}")
    
    vqe_qiskit.plot_results()

    print("\n" + "=" * 60)
    print("VQE Optimization - Ising - trivial-ansatz")
    print("=" * 60)


    # Ising for 3 spins

    J_sys = np.array([
        [0.0,  1.0,  0.0],
        [1.0,  0.0,  0.8],
        [0.0,  0.8,  0.0],
    ], dtype=float)

    h_sys = np.array([0.2, -0.1, 0.05], dtype=float)

    vqe_ising_triv = VQE_Ising_triv_FiniteDiff_ConstStep(
        max_iter=500,
        learning_rate=0.01,
        gradient_eps=1e-6,
        store_history=True,
        J=J_sys,
        h=h_sys
    )
    
    dim_theta=J_sys.shape[0]
    theta=np.zeros(dim_theta)
    for i in range(dim_theta):
            theta[i]=np.pi/dim_theta*i

    optimal_theta_qk, optimal_energy_qk = vqe_ising_triv.run(initial_params=theta)
    

    print(f"\nOptimal θ: {optimal_theta_qk}")
    print(f"Optimal Energy: {optimal_energy_qk:.6f}")
    print(f"Theoretical minimum: {-1.95:.6f}")

    vqe_ising_triv.plot_results()


    print("\n" + "=" * 60)
    print("VQE Optimization - Ising - real-ansatz")
    print("=" * 60)


    
    J_sys = np.array([
        [0.0,  1.0,  0.0],
        [1.0,  0.0,  0.8],
        [0.0,  0.8,  0.0],
    ], dtype=float)

    h_sys = np.array([0.2, -0.1, 0.05], dtype=float)

    reps_sys=0

    vqe_ising_real = VQE_Ising_real_FiniteDiff_ConstStep(
        max_iter=500,
        learning_rate=0.01,
        gradient_eps=1e-6,
        store_history=True,
        J=J_sys,
        h=h_sys,
        reps=reps_sys
    )
    dim_theta=(reps_sys+1)*J_sys.shape[0]
    theta=np.zeros(dim_theta)
    for i in range(dim_theta):
        theta[i]=np.pi/dim_theta*i

    optimal_theta_qk, optimal_energy_qk = vqe_ising_real.run(initial_params=theta)
    

    print(f"\nOptimal θ: {optimal_theta_qk}")
    print(f"Optimal Energy: {optimal_energy_qk:.6f}")
    print(f"Theoretical minimum: {-1.95:.6f}")

    vqe_ising_real.plot_results()



    print("\n" + "=" * 60)
    print("VQE Optimization - Ising - complex-ansatz")
    print("=" * 60)


    
    J_sys = np.array([
        [0.0,  1.0,  0.0],
        [1.0,  0.0,  0.8],
        [0.0,  0.8,  0.0],
    ], dtype=float)

    h_sys = np.array([0.2, -0.1, 0.05], dtype=float)

    reps_sys=0

    vqe_ising_complex = VQE_Ising_complex_FiniteDiff_ConstStep(
        max_iter=500,
        learning_rate=0.01,
        gradient_eps=1e-6,
        store_history=True,
        J=J_sys,
        h=h_sys,
        reps=reps_sys
    )
    dim_theta=(reps_sys+1)*J_sys.shape[0]*2
    theta=np.zeros(dim_theta)
    for i in range(dim_theta):
        theta[i]=np.pi/dim_theta*i

    optimal_theta_qk, optimal_energy_qk = vqe_ising_complex.run(initial_params=theta)
    

    print(f"\nOptimal θ: {optimal_theta_qk}")
    print(f"Optimal Energy: {optimal_energy_qk:.6f}")
    print(f"Theoretical minimum: {-1.95:.6f}")

    vqe_ising_complex.plot_results()



#to do:
#inital theta (random, several)
#plots: energy, theta
#opti (grad, adam, natural)




