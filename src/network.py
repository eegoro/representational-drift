import torch


class HebbianNetwork:
    def __init__(
        self,
        N: int = 10,
        dt: float = 10.0,
        tau_r: float = 10.0,
        tau_W: float = 2000.0,
        tau_h: float = 500.0,
        eta: float = 0.005,
        lam: float = 0.1,
        kappa_h: float = 0.1,
        m_target: float = 0.5,
        W_scale: float = 0.1,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.N = N
        self.dt = dt
        self.tau_r = tau_r
        self.tau_W = tau_W
        self.tau_h = tau_h
        self.eta = eta
        self.lam = lam
        self.kappa_h = kappa_h
        self.m_target = m_target
        self.device = torch.device(device)

        torch.manual_seed(seed)
        # All-negative structural baseline (inhibitory)
        W_struct = -torch.rand(N, N, device=self.device) * (W_scale / N)
        W_struct.fill_diagonal_(0.0)
        self.W_struct = W_struct

        self.reset()

    def reset(self):
        self.r = torch.zeros(self.N, device=self.device)
        self.W = self.W_struct.clone()
        self.h = torch.tensor(0.0, device=self.device)
        self.epsilon = torch.zeros(self.N, device=self.device)

    @staticmethod
    def phi(x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)

    def step(self, I_ext: torch.Tensor, noise_std: float = 0.0, homeostasis: bool = False):
        """Simultaneous Euler update of r and W using state at time t."""
        phi_r = self.phi(self.r)
        dr = (-self.r + self.W @ phi_r + I_ext + self.epsilon + self.h) / self.tau_r
        dW = (
            -self.lam * (self.W - self.W_struct)
            + self.eta * torch.outer(phi_r, phi_r)
        ) / self.tau_W
        if noise_std > 0.0:
            dW = dW + noise_std * torch.randn_like(dW) / self.tau_W
        self.r = self.r + self.dt * dr
        self.W = self.W + self.dt * dW
        self.W.fill_diagonal_(0.0)
        if homeostasis:
            m_mean = self.r.mean()
            dh = self.kappa_h * (self.m_target - m_mean) / self.tau_h
            self.h = self.h + self.dt * dh


# Backward-compatibility alias
RateRNN = HebbianNetwork
