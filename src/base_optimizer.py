import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector
from typing import Optional, List, Tuple, Any
from qiskit.circuit.library import RealAmplitudes
from qiskit.circuit.library import EfficientSU2

from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock




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
        eps_energy: float = 1e-6,
        store_history: bool = True,
        random_state: Optional[int] = None
    ):
        self.max_iter = max_iter
        self.eps_energy = eps_energy
        self.random_state = random_state
        self.params= None
        self.iteration=1
        self.state=None
        self.params_dim=None
        
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
        Berechnet den Gradienten bezüglich der Parameter.
        Muss in Subklassen implementiert werden.
        """
        pass
    
    def compute_expectation_value(self, state, operator) -> float:
        """E(θ) = ⟨ψ(θ)|operaor|ψ(θ)⟩"""
        return np.real(np.vdot(state.data, operator.data @ state.data))
    
    def get_state(self, params) -> Statevector:
        return self.initial_state.evolve(self.ansatz(params))
    
    def step_size(self) -> float:
        """
        Learning rate schedule. Kann überschrieben werden.
        Default: konstant 0.01
        """
        return 0.01
    
    def _update(self, gradient: Any) -> Any:
        """
        Standard Gradientenabstieg: θ_new = θ_old - η * ∇E
        """
        eta = self.step_size()
        return self.params - eta * gradient
    
    def _store_iteration(self):
        """Speichert Historie falls aktiviert"""
        if self.store_history_flag:
            self.history_params.append(self.params)
            self.history_energy.append(self.compute_expectation_value(self.state,self.hamilton))
            self.history_state.append(self.state)
    
    def run(self, initial_params: Any) -> Tuple[Any, float]:
        """
        Führt die Optimierung aus.
        
        Returns:
            (optimale_parameter, optimale_energie)
        """
        self.params_dim=initial_params.size
        self.params=initial_params
        
        for self.iteration in range(self.max_iter):
            print(self.iteration/self.max_iter*100,'%')
            self.state=self.get_state(self.params)
            self._store_iteration()
            
            gradient = self.compute_gradient()
            new_params = self._update(gradient)
            new_state= self.get_state(new_params)

            E_old, E_new = self.compute_expectation_value(self.state,self.hamilton), self.compute_expectation_value(new_state,self.hamilton)
            
            if np.abs(E_new - E_old) < self.eps_energy:
                self.params = new_params
                break
            
            self.params = new_params
        
        self.state=self.get_state(self.params)
        self._store_iteration()
        final_energy = self.compute_expectation_value(self.state,self.hamilton)
        return self.params, final_energy
    
    def plot_results(self):
        """Plottet die Optimierungshistorie"""
        if not self.store_history_flag:
            print("No history stored. Set store_history=True to enable plotting.")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
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
        
        # # Plot 2: Zustandsentwicklung
        # ax2 = axes[1]
        # if self.history_state and hasattr(self.history_state[0], '__iter__'):
        #     states_array = np.array([s if isinstance(s, np.ndarray) else s.data 
        #                            for s in self.history_state])
            
        #     if states_array.ndim > 1 and states_array.shape[1] >= 2:
        #         ax2.plot(np.abs(states_array[:, 0])**2, label='|⟨0|ψ⟩|²', linewidth=2)
        #         ax2.plot(np.abs(states_array[:, 1])**2, label='|⟨1|ψ⟩|²', linewidth=2)
        #         ax2.set_xlabel('Iteration')
        #         ax2.set_ylabel('Probability')
        #         ax2.set_title('State Evolution')
        #         ax2.legend()
        #         ax2.grid(True, alpha=0.3)
        
        # plt.tight_layout()
        # plt.show()
        


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
    
    def compute_gradient(self) -> Any:
        """Numerischer Gradient: (E(θ+ε) - E(θ)) / ε"""
        gradient=np.zeros(self.params_dim, dtype=float)
        for i in range(self.params_dim):
            energy_current = self.compute_expectation_value(self.state,self.hamilton)
            params_shifted=self.params.copy()
            params_shifted[i]=self.params[i]+self.gradient_eps
            state_shifted=self.get_state(params_shifted)
            energy_shifted = self.compute_expectation_value(state_shifted,self.hamilton)
            gradient[i]=(energy_shifted-energy_current)/self.gradient_eps
        return gradient



class PSR_Gradient:
    """
    Mixin für numerische Gradientenberechnung via finite Differenzen.
    """
    
    def compute_gradient(self) -> Any:
        """Numerischer Gradient: (E(θ+pi/2) - E(θ-p/2)) / 2"""
        gradient=np.zeros(self.params_dim, dtype=float)
        for i in range(self.params_dim):
            params_plus=self.params.copy()
            params_plus[i]=self.params[i]+np.pi/2
            state_plus=self.get_state(params_plus)
            params_minus=self.params.copy()
            params_minus[i]=self.params[i]-np.pi/2
            state_minus=self.get_state(params_minus)
            energy_plus = self.compute_expectation_value(state_plus,self.hamilton)
            energy_minus = self.compute_expectation_value(state_minus,self.hamilton)
            gradient[i]=(energy_plus-energy_minus)/2
        return gradient
    

class SPSA_Gradient:
    def __init__(self, *args, gradient_eps: float = 1e-6, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_eps = gradient_eps
    
    def compute_gradient(self) -> Any:
        """Simultaneous Perturbation Stochastic Approximation:
            ( f(θ + s_k·Δ_k) - f(θ - s_k·Δ_k) ) / (2·s_k)  ·  Δ_k
        """
        delta_k=(2 * np.random.randint(0, 2, size=self.params_dim) - 1)

        params_plus=self.params+self.gradient_eps*delta_k
        state_plus=self.get_state(params_plus)
        params_minus=self.params-self.gradient_eps*delta_k
        state_minus=self.get_state(params_minus)
        energy_plus = self.compute_expectation_value(state_plus,self.hamilton)
        energy_minus = self.compute_expectation_value(state_minus,self.hamilton)

        gradient=(energy_plus-energy_minus)/(2*self.gradient_eps)*delta_k
        return gradient    


# ============================================================================
# STEP SIZE STRATEGIES / OPTIMIZERS
# ============================================================================

class ConstantStepSize:
    """Konstante Lernrate"""
    def __init__(self, *args, learning_rate: float = 0.01, **kwargs):
        self.learning_rate = learning_rate
        super().__init__(*args, **kwargs)
    
    def step_size(self) -> float:
        return self.learning_rate


class DecayingStepSize:
    """Abnehmende Lernrate: η_k = η_0 / (1 + decay * k)"""
    def __init__(self, *args, learning_rate: float = 0.1, decay: float = 0.01, **kwargs):
        self.learning_rate = learning_rate
        self.decay = decay
        super().__init__(*args, **kwargs)
    
    def step_size(self) -> float:
        return self.learning_rate * (1.0 + self.decay * self.iteration)


class Adam:
    """Adam optimizer"""
    def __init__(self, beta1=0.9, beta2=0.999, eps=1e-8, learning_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.eta = learning_rate
        self.m = None
        self.v = None

    def _update(self, gradient: Any) -> Any:
        # Lazy state initialization (handles shape (n_dim,) or (n_batch, n_dim))
        if self.m is None or self.v is None:
            self.m = np.zeros_like(self.params, dtype=float)
            self.v = np.zeros_like(self.params, dtype=float)
            self.iteration+=1

        # Exponential moving averages
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (gradient * gradient)

        # Bias corrections based on exponential moving averages
        m_hat = self.m / (1.0 - self.beta1 ** self.iteration)
        v_hat = self.v / (1.0 - self.beta2 ** self.iteration)

        # Parameter update
        params_new = self.params - self.eta * m_hat / (np.sqrt(v_hat) + self.eps)
        return params_new
    



class qng_finite_difference:

    def __init__(self, *args, qng_eps: float = 1e-6, **kwargs):
        self.qng_eps = qng_eps
        super().__init__(*args, **kwargs)

    def _update(self, gradient: Any) -> Any:
        """
        Standard Gradientenabstieg: θ_new = θ_old - η * ∇E
        """
        fsm = self.fsm()
        return self.params - self.step_size()*fsm @ gradient
    
    def fsm(self):
        g = np.zeros((self.params_dim,self.params_dim))
        Id=np.identity(self.params_dim, dtype=float)
        state=self.state.data
        for i in range(self.params_dim):
            state_i=self.state_deriv_data(i)
            state_i_state=np.vdot(state_i, state)
            for j in range(self.params_dim):
                state_j=self.state_deriv_data(j)
                state_i_state_j=np.vdot(state_i, state_j)
                state_state_j=np.vdot(state, state_j)
                g[i][j]=np.real(state_i_state_j-state_i_state*state_state_j)
#nebendiag reichen
#plus minus eps
        return np.linalg.pinv(g)
        
    
    def state_deriv_data(self,i):
        params_shifted=self.params.copy()
        params_shifted[i]=self.params[i]+self.qng_eps
        state_shifted=self.get_state(params_shifted)
        state_data=self.state.data
        state_shifted_data=state_shifted.data
        return (state_shifted_data-state_data)/self.qng_eps






class qng_bda:
    def _update(self, gradient: Any) -> Any:
        """
        Standard Gradientenabstieg: θ_new = θ_old - η * ∇E
        """
        fsm = self.fsm()
        #print('tessst:',self.params,fsm,gradient)
        return self.params - self.step_size()*fsm @ gradient
    
    def fsm(self):
        g = np.zeros((self.params_dim,self.params_dim))
        # here should be the code
        

    #inital operator
    #determine block
        #determine phi_l
            #get teil ansatz
                #get variables
        #get pauli*pauli / pauli
        #determine expec.
        #final block formel
    #put it together
    #invert it


# ============================================================================
# ANSATZ
# ============================================================================


class triv_ansatz:
    def ansatz(self,theta):
        u=np.array([[1]])
        for i in range(self.dim):
            u=np.kron(u,np.array([[np.cos(theta[i]/2), -np.sin(theta[i]/2)],[np.sin(theta[i]/2),  np.cos(theta[i]/2)]]))
        return Operator(u)


class real_ansatz:
    def __init__(self,reps,**kwargs):
        super().__init__(**kwargs)
        self.reps=reps

    def ansatz(self,theta):
        num_qubits = self.dim
        reps = self.reps
        entanglement = 'full'

        ansatz = RealAmplitudes(num_qubits=num_qubits,reps=reps,entanglement=entanglement,insert_barriers=True,).decompose()
        ansatz_bound = ansatz.assign_parameters(theta)
        return ansatz_bound
    

    

class complex_ansatz:
    def __init__(self,reps,**kwargs):
        super().__init__(**kwargs)
        self.reps=reps

    def ansatz(self,theta):
        num_qubits = self.dim
        reps = self.reps
        entanglement = 'full'

        ansatz = EfficientSU2(num_qubits=num_qubits,reps=reps,entanglement=entanglement,insert_barriers=True).decompose()
        ansatz_bound = ansatz.assign_parameters(theta)
        return ansatz_bound


# ============================================================================
# QUANTUM SYSTEMS
# ============================================================================

class OneQubitSystem(BaseOptimizer):
    """
    Abstrakte Basisklasse für Ein-Qubit-Systeme.
    Definiert gemeinsame Struktur für numpy und Qiskit Implementierungen.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dim=1
        self._setup_hamiltonian()
        self._setup_initial_state()
    


    def _setup_hamiltonian(self):
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        self.hamilton = Operator(-X - Z)
    
    def _setup_initial_state(self):
        qc = QuantumCircuit(1)
        self.initial_state = Statevector.from_instruction(qc)

    # def ansatz(self, theta) -> Statevector:
    #     """U(θ) = exp(-iθY/2)"""
    #     c = np.cos(theta[0] / 2)
    #     s = np.sin(theta[0] / 2)
    #     U_matrix = np.array([[c, s], [-s, c]], dtype=complex)
    #     return Operator(U_matrix)
    
    

class n_dim_ising(BaseOptimizer):
    def __init__(self,J,h,**kwargs):
        super().__init__(**kwargs)
        self.dim=J.shape[0]
        self._setup_hamiltonian(J,h)
        self._setup_initial_state()

    def _setup_hamiltonian(self,J,h):
        hamilton=np.zeros((2**self.dim,2**self.dim))
        for i in range(self.dim):
            hamilton=hamilton-h[i]*self.pauli_i_j(i,i)
            for j in range(i+1,self.dim):
                hamilton=hamilton-J[i,j]*self.pauli_i_j(i,j)
        self.hamilton=Operator(hamilton)
    
    def _setup_initial_state(self):
        self.initial_state=Statevector.from_instruction(QuantumCircuit(self.dim))

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
    

class hydrogen(BaseOptimizer):
    def __init__(self,dist,**kwargs):
        super().__init__(**kwargs)
        self.dim=4
        self._setup_hamiltonian(dist)
        self._setup_initial_state()

    def _setup_hamiltonian(self,dist):
        driver = PySCFDriver(
            atom=f"H 0 0 {-dist/2}; H 0 0 {dist/2}",  # symmetric around origin
            basis="sto3g",
            charge=0,
            spin=0,
            unit=DistanceUnit.ANGSTROM,
        )
        problem = driver.run()

        # electronic (fermionic) Hamiltonian in second quantization
        second_q_op = problem.hamiltonian.second_q_op()

        # 2) Map to qubits using Jordan–Wigner (no two-qubit reduction)
        self.mapper = JordanWignerMapper()
        qubit_op = self.mapper.map(second_q_op)   # SparsePauliOp

        # 3) Convert to matrix and add nuclear repulsion
        H_el = qubit_op.to_matrix()          # 16x16 electronic part
        enuc = problem.nuclear_repulsion_energy
        self.hamilton = H_el + enuc * np.eye(H_el.shape[0], dtype=complex)


    
    def _setup_initial_state(self):
        
        hf_circuit = HartreeFock(
            num_spatial_orbitals=2,
            num_particles=(1, 1),
            qubit_mapper=self.mapper,
        )
        
        self.initial_state = Statevector.from_instruction(hf_circuit)

        #self.initial_state=Statevector.from_instruction(QuantumCircuit(self.dim))




# ============================================================================
# KONKRETE VQE IMPLEMENTIERUNGEN (durch Mixins komponiert)
# ============================================================================

class VQE_one_qubit_FiniteDiff_ConstStep(
    FiniteDifferenceGradient,
    ConstantStepSize,
    triv_ansatz,
    OneQubitSystem
):
    """VQE mit NumPy, finiten Differenzen und konstanter Schrittweite"""
    pass


class VQE_one_qubit_PSR_ConstStep_qng(
    PSR_Gradient,
    DecayingStepSize,
    qng_finite_difference,
    real_ansatz,
    OneQubitSystem
):
    """VQE mit NumPy, finiten Differenzen und konstanter Schrittweite"""
    pass


class VQE_one_qubit_PSR_ConstStep(
    PSR_Gradient,
    Adam,
    real_ansatz,
    OneQubitSystem
):
    """VQE mit NumPy, finiten Differenzen und konstanter Schrittweite"""
    pass



class VQE_Ising_triv_PSR_ConstStep(
    PSR_Gradient,
    ConstantStepSize,
    triv_ansatz,
    n_dim_ising
):
    """VQE mit NumPy, finiten Differenzen und konstanter Schrittweite"""
    pass


class VQE_Ising_real_FiniteDiff_ConstStep(
    FiniteDifferenceGradient,
    ConstantStepSize,
    real_ansatz,
    n_dim_ising
):
    """VQE mit NumPy, finiten Differenzen und konstanter Schrittweite"""
    pass


class VQE_Ising_real_PSR_qng(
    PSR_Gradient,
    DecayingStepSize,
    qng_finite_difference,
    real_ansatz,
    n_dim_ising
):
    """VQE mit NumPy, finiten Differenzen und konstanter Schrittweite"""
    pass


class VQE_Ising_real_PSR_Adam(
    PSR_Gradient,
    Adam,
    real_ansatz,
    n_dim_ising
):
    """VQE mit NumPy, finiten Differenzen und konstanter Schrittweite"""
    pass

class VQE_hydrogen_real_PSR_qng(
    PSR_Gradient,
    DecayingStepSize,
    qng_finite_difference,
    real_ansatz,
    hydrogen
):
    """VQE mit NumPy, finiten Differenzen und konstanter Schrittweite"""
    pass


# ============================================================================
# Example
# ============================================================================

if __name__ == "__main__":
    # print("=" * 60)
    # print("VQE Optimization - one qubit (VQE_one_qubit_PSR_ConstStep)")
    # print("=" * 60)

#     vqe_one_qubit_PSR = VQE_one_qubit_PSR_ConstStep(
#         max_iter=500,
#         learning_rate=0.01,
#         store_history=True,
#         eps=1e-6,
#         reps=0
#     )
    
#     optimal_theta, optimal_energy = vqe_one_qubit_PSR.run(initial_params=np.array([np.pi/2]))
    
#     print(f"\nOptimal θ: {optimal_theta}")
#     print(f"Optimal Energy: {optimal_energy:.6f}")
#     print(f"Theoretical minimum: {-np.sqrt(2):.6f}")
    
#     vqe_one_qubit_PSR.plot_results()
    



#     print("=" * 60)
#     print("VQE Optimization - one qubit (VQE_one_qubit_PSR_ConstStep_qng)")
#     print("=" * 60)

#     vqe_one_qubit_PSR = VQE_one_qubit_PSR_ConstStep_qng(
#         max_iter=500,
#         #learning_rate=0.01,
#         store_history=True,
#         qng_eps=1e-6,
#         reps=0
#     )
    
#     optimal_theta, optimal_energy = vqe_one_qubit_PSR.run(initial_params=np.array([np.pi/2]))
    
#     print(f"\nOptimal θ: {optimal_theta}")
#     print(f"Optimal Energy: {optimal_energy:.6f}")
#     print(f"Theoretical minimum: {-np.sqrt(2):.6f}")
    
#     vqe_one_qubit_PSR.plot_results()

# #--------------------------------------------------------------------------------------------------------


#     print("\n" + "=" * 60)
#     print("VQE Optimization - Ising (VQE_Ising_real_PSR_qng)")
#     print("=" * 60)

#     J_sys = np.array([
#         [0.0,  1.0,  0.0],
#         [1.0,  0.0,  0.8],
#         [0.0,  0.8,  0.0],
#     ], dtype=float)

#     h_sys = np.array([0.2, -0.1, 0.05], dtype=float)

#     reps_sys=1

#     vqe_ising_real = VQE_Ising_real_PSR_qng(
#         max_iter=200,
#         #learning_rate=0.01,
#         store_history=True,
#         reps=reps_sys,
#         J=J_sys,
#         h=h_sys
#     )
    
#     dim_theta=(reps_sys+1)*J_sys.shape[0]
#     theta=np.zeros(dim_theta)
#     for i in range(dim_theta):
#         theta[i]=np.pi/dim_theta*i

#     optimal_theta, optimal_energy = vqe_ising_real.run(initial_params=theta)

#     print(f"\nOptimal θ: {optimal_theta}")
#     print(f"Optimal Energy: {optimal_energy:.6f}")
#     print(f"Theoretical minimum: {-1.95:.6f}")

#     vqe_ising_real.plot_results()




#     print("\n" + "=" * 60)
#     print("VQE Optimization - Ising - (VQE_Ising_real_PSR_Adam)")
#     print("=" * 60)
    
#     J_sys = np.array([
#         [0.0,  1.0,  0.0],
#         [1.0,  0.0,  0.8],
#         [0.0,  0.8,  0.0],
#     ], dtype=float)

#     h_sys = np.array([0.2, -0.1, 0.05], dtype=float)

#     reps_sys=1

#     vqe_ising_real_adam_SPSA = VQE_Ising_real_PSR_Adam(
#         max_iter=500,
#         learning_rate=0.1,
#         #gradient_eps=1e-6,
#         store_history=True,
#         reps=reps_sys,
#         J=J_sys,
#         h=h_sys
#     )
#     dim_theta=(reps_sys+1)*J_sys.shape[0]
#     theta=np.zeros(dim_theta)
#     for i in range(dim_theta):
#         theta[i]=np.pi/dim_theta*i

#     optimal_theta, optimal_energy = vqe_ising_real_adam_SPSA.run(initial_params=theta)
    

#     print(f"\nOptimal θ: {optimal_theta}")
#     print(f"Optimal Energy: {optimal_energy:.6f}")
#     print(f"Theoretical minimum: {-1.95:.6f}")

#     vqe_ising_real_adam_SPSA.plot_results()




    #---------------------------------------------------------------------------------------

    




    

    print("\n" + "=" * 60)
    print("VQE Optimization - Ising (VQE_hydrogen_real_PSR_qng)")
    print("=" * 60)


    hist=[]
    d_values = np.linspace(0.15, 4.0, 30)   # in Å
    for d_sys in d_values:
        print(d_sys)
        dim=4
        reps_sys=0

        vqe_hydrogen_real = VQE_hydrogen_real_PSR_qng(
            max_iter=200,
            learning_rate=0.01,
            store_history=True,
            reps=reps_sys,
            dist=d_sys
        )




        dim_theta=(reps_sys+1)*dim
        theta=np.zeros(dim_theta)
        for i in range(dim_theta):
            theta[i]=np.pi/dim_theta*i

        optimal_theta, optimal_energy = vqe_hydrogen_real.run(initial_params=theta)

        print(f"\nOptimal θ: {optimal_theta}")
        print(f"Optimal Energy: {optimal_energy:.6f}")
        print(f"Theoretical minimum: {-1.95:.6f}")

        hist.append(optimal_energy)
        #vqe_hydrogen_real.plot_results()

    #print(hist)
    plt.plot(d_values, hist, "o-")
    plt.xlabel("H-H distance (Å)")
    plt.ylabel("Energy (Hartree)")
    plt.grid(True)
    plt.show()



# to do:

#   gute plots (inital theta (random, mehrere))
#   opti (quantum grad estimation, qng real, via phase estimation E bestimmen, coolen opti)
#   one qubit with generall ansatz?                                                                         check
#   H2
#   gute ordner struktur


#check if qng actually works(faster/ising)      check
#do step generall                               check
#eps fixen                                      check


