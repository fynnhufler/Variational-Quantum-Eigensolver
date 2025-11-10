# Building a Variational Quantum Eigensolver

## Description:
We are planning to implement a Variational Quantum Eigensolver that approximates ground state eigenstates and eigenvalues of simple Hamiltonians like the Hydrogen Atom or Spin Glass Model. We aim to design a modular system for performing variational optimization on quantum systems, allowing the use of different optimizers and Hamiltonian models.

## Planned Directory Structure:
- `demo.ipynb`: Jupyter notebook showcasing minimal examples of optimizers, system initializations, and visualizations.
- `src/`: Contains all class implementations and core logic.
  - `base_optimizer.py`: Base class for optimizer implementations.
  - `optimizers/`: Folder for different optimizer classes (e.g., `natural_gradient.py`, `adam.py`).
  - `systems/`: Folder for system classes like `IsingModel.py`, `HydrogenHamiltonian.py`.
- `results/`: Folder for storing plots, visualizations, and simulation results.

## Chronological Steps:
1. **Project structure**: Initialize directory structure with folders `src/`, `results/`, and a sample `demo.ipynb`.
2. **Implement BaseOptimizer class**: Create a general `BaseOptimizer` class that takes in Hamiltonians, neural network layers, and different optimization methods (e.g., natural gradient, Adam).
3. **Develop Optimizer subclasses**: Implement the specific optimization algorithms as subclasses of `BaseOptimizer`, e.g., `NaturalDescent` and `Adam`.
4. **Implement System Initialization classes**: Create subclasses like `IsingModel` and `HydrogenHamiltonian`, which define the specific Hamiltonians for different quantum systems.
5. **Run simulations**: Each team member will choose a quantum system and implement the corresponding simulation, running optimizations and observing performance.
6. **Visualization and results**: Use the `demo.ipynb` to demonstrate optimizer performance and visualize results. Store and organize results in the `results/` folder.

## Planned Contributions from each member:
Each member will focus on implementing at least one class of their own, and running at least one simulation for a (physcial) system of their choice.

## Notes & References:
  - [Quantum Natural Gradient](https://arxiv.org/abs/1909.02108)
  - [A variational eigenvalue solver on a quantum processor](https://arxiv.org/abs/1304.3061)
  - [An Overview of Variational Quantum Algorithms](https://www.youtube.com/watch?v=SU4FG2eT1rI&t=72s)
  - [Qiskit Summer School 2021 on VQE's](https://github.com/Qiskit/platypus/blob/main/notebooks/summer-school/2021/resources/lab-notebooks/lab-2.ipynb)
  - [A Comparative Analysis on of Classical and Quantum Optimization Methods](https://arxiv.org/abs/2412.19176)

