# ============================================================================
# QUANTUM SYSTEMS
# ============================================================================


from base_optimizer import *


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