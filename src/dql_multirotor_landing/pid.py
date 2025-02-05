from typing import Any

import torch
import torch.nn.functional as F


class PIDController:
    """Pid controller that controls `height` and `yaw`"""

    def __init__(
        self,
        kp: Any = [4, 8],
        ki: Any = [1, 1],
        kd: Any = [1, 4],
        set_point: Any = [0, 0],
        device="cpu",
        *,
        windup_max=10,
        sample_time=0.02,
    ):
        self.device = device
        self.kp = torch.tensor(kp).to(self.device)
        self.ki = torch.tensor(ki).to(self.device)
        self.kd = torch.tensor(kd).to(self.device)
        self.set_point = torch.tensor(set_point).to(self.device)
        self.windup_max = torch.tensor(windup_max).to(self.device)

        self.p_term = torch.zeros_like(self.kp).to(self.device).float()
        self.i_term = torch.zeros_like(self.kp).to(self.device).float()
        self.d_term = torch.zeros_like(self.kp).to(self.device).float()

        self.sample_time = sample_time
        self.last_error = torch.zeros_like(self.kp).to(self.device)
        self.last_y = torch.tensor(set_point).to(self.device)

    def output(self, y_measured):
        y_measured = torch.tensor(y_measured).to(self.device)
        # Compute the current error
        error = self.set_point - y_measured

        # Get the PID values
        self.p_term = self.kp * error

        # Compute the integral term
        self.i_term += error * self.sample_time
        I_out = self.ki * self.i_term

        # Compute the derivative term
        self.d_term = (error - self.last_error) / self.sample_time
        D_out = self.kd * self.d_term

        # TODO: Understand if it's needed.
        # Anti-windup
        # self.i_term = torch.clamp(self.i_term, -self.windup_max, self.windup_max)

        # Salvataggio stati
        self.last_error = error
        self.last_y = y_measured
        result = self.p_term + I_out + D_out
        limits = torch.tensor([[0, 2], [-1, 1]])
        result = torch.clamp(result, min=limits[:, 0], max=limits[:, 1])
        return result

    def reset(self):
        self.p_term = torch.zeros_like(self.kp).to(self.device).float()
        self.i_term = torch.zeros_like(self.ki).to(self.device).float()
        self.d_term = torch.zeros_like(self.kd).to(self.device).float()
        self.last_error = torch.zeros_like(self.kp).to(self.device)
        self.last_y = torch.zeros_like(self.kp).to(self.device)

    def set_setpoint(self, set_point):
        self.set_point = torch.tensor(set_point).to(self.device)


a, b = PIDController().output(([10, 10]))
print(a.item(), b.item())
