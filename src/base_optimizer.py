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

    @abstractmethod
    def _update(self, gradient: Any) -> Any:
        """
        Compute update
        """
        pass
    
    def compute_expectation_value(self, state: Statevector, operator: Operator) -> float:
        """E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩"""
        return np.real(np.vdot(state.data, operator.data @ state.data))
    
    def get_state(self, params: np.ndarray) -> Statevector:
        """Get quantum state for given parameters"""
        return self.initial_state.evolve(self.ansatz(params))
    

    
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


















    #doc strings schreiben 
    #gradient descent auslagern
    #ordner structur anlegen                check
    #code durchgehen und schöner machen
    #plots entfernen in main                check
    #optional demo schön machen
    #nur wichtige VQE behalten
