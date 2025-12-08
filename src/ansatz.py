# ============================================================================
# ANSATZ CLASSES
# ============================================================================


from base_optimizer import *

class TrivialAnsatz:
    """Single-qubit Ry rotation ansatz applied independently to each qubit."""
    def ansatz(self, theta: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.dim)
        for i in range(self.dim):
            qc.ry(theta[i], i)
        return qc


class RealAmplitudesAnsatz:
    """Hardware-efficient ansatz based on Qiskit's RealAmplitudes template."""
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
    """Hardware-efficient ansatz ased on Qiskit's EfficientSU2 template."""
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
