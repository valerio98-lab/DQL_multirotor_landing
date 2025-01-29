from typing import Any

import torch


class PIDController:
    def __init__(
        self,
        kp: Any = [5, 8],
        ki: Any = [10, 1],
        kd: Any = [0, 0],
        set_point: Any = [0, 0],
        device="cpu",
        *,
        windup_max=10,
        sample_time=0.016,
    ):
        self.device = device
        self.kp = torch.tensor(kp).to(self.device)
        self.ki = torch.tensor(ki).to(self.device)
        self.kd = torch.tensor(kd).to(self.device)
        self.set_point = set_point
        self.windup_max = torch.tensor(windup_max).to(self.device)

        self.p_term = torch.zeros_like(self.kp).to(self.device)
        self.i_term = torch.zeros_like(self.kp).to(self.device)
        self.d_term = torch.zeros_like(self.kp).to(self.device)

        self.sample_time = sample_time
        self.last_error = torch.zeros_like(self.kp).to(self.device)
        self.last_y = torch.zeros_like(self.kp).to(self.device)

    def output(self, y_measured):
        # Compute the current error
        error = self.set_point - torch.tensor(y_measured).to(self.device)

        # Get the PID values
        self.p_term = self.kp * error
        self.d_term = self.kd * (self.last_y - y_measured) / self.sample_time
        self.i_term += self.ki * error * self.sample_time

        # TODO: Understand if it's needed.
        # Anti-windup
        # self.i_term = torch.clamp(self.i_term, -self.windup_max, self.windup_max)

        # Salvataggio stati
        self.last_error = error
        self.last_y = y_measured

        return self.p_term + self.i_term + self.d_term
