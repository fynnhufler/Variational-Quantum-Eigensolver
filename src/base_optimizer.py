import numpy as np

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