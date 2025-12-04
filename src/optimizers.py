# ============================================================================
# STEP SIZE STRATEGIES
# ============================================================================


from base_optimizer import *



class GradientDescent:
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    def _update(self, gradient: Any) -> Any:
        """
        Standard Gradientenabstieg: θ_new = θ_old - η * ∇E
        """
        eta = self.step_size()
        return self.params - eta * gradient

class QuantumNaturalGradient:

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
        return np.linalg.pinv(g)
        
    
    def state_deriv_data(self,i):
        params_shifted=self.params.copy()
        params_shifted[i]=self.params[i]+self.qng_eps
        state_shifted=self.get_state(params_shifted)
        state_data=self.state.data
        state_shifted_data=state_shifted.data
        return (state_shifted_data-state_data)/self.qng_eps



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

