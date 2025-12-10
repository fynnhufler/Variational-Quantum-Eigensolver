# ============================================================================
# VQE IMPLEMENTATIONS
# ============================================================================

from base_optimizer import *
from ansatz import *
from gradients import *
from optimizers import *
from systems import *

#SIngle Qubit Systems

class VQE_OneQubit_FiniteDiff_Const(
    FiniteDifferenceGradient,
    GradientDescent,
    ConstantStepSize,
    TrivialAnsatz,
    OneQubitSystem
):
    """Single qubit VQE with finite differences and constant step size"""
    pass

class VQE_OneQubit_FiniteDiff_Const_Real(
    FiniteDifferenceGradient,
    GradientDescent,
    ConstantStepSize,
    RealAmplitudesAnsatz,
    OneQubitSystem
):
    """Single qubit VQE with finite differences, RealAmplitudes ansatz"""
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
    QuantumNaturalGradient,
    DecayingStepSize,
    RealAmplitudesAnsatz,
    IsingModel
):
    """Ising model VQE with Quantum Natural Gradient Descent"""
    pass


class VQE_Ising_PSR_Adam(
    ParameterShiftGradient,
    Adam,
    RealAmplitudesAnsatz,
    IsingModel
):
    """Ising model VQE with SPSA (efficient for high dimensions) and Adam"""
    pass

class VQE_Ising_PSR_const(
    ConstantStepSize,
    ParameterShiftGradient,
    GradientDescent,
    RealAmplitudesAnsatz,
    IsingModel
):
    """VQE_Ising_PSR_const"""
    pass


#------------------------------------------------gradient-------------------------

class VQE_Ising_QNG_finit(
    FiniteDifferenceGradient,
    QuantumNaturalGradient,
    DecayingStepSize,
    RealAmplitudesAnsatz,
    IsingModel
):
    """Ising model VQE with Quantum Natural Gradient Descent"""
    pass


class VQE_Ising_QNG_spsa(
    SPSAGradient,
    QuantumNaturalGradient,
    DecayingStepSize,
    RealAmplitudesAnsatz,
    IsingModel
):
    """Ising model VQE with SPSA (efficient for high dimensions) and Adam"""
    pass

class VQE_Ising_QNG_psr(
    ParameterShiftGradient,
    QuantumNaturalGradient,
    DecayingStepSize,
    RealAmplitudesAnsatz,
    IsingModel
):
    """VQE_Ising_PSR_const"""
    pass





# H-Molecule

class VQE_H2_QNG(
    ParameterShiftGradient,
    QuantumNaturalGradient,
    DecayingStepSize,
    RealAmplitudesAnsatz,
    HydrogenMolecule
):
    """H2 molecule VQE with Quantum Natural Gradient Descent"""
    pass

class VQE_H2_QNG_comp(
    SPSAGradient,
    QuantumNaturalGradient,
    DecayingStepSize,
    ComplexAnsatz,
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



