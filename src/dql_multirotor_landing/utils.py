from dataclasses import dataclass

import torch


@dataclass
class Action:
    thrust: float
    roll: float
    pitch: float
    yaw: float
    left_wheel: float
    right_wheel: float

    def to_tensor(self, device):
        return torch.tensor(
            [
                [
                    self.thrust,
                    self.roll,
                    self.pitch,
                    self.yaw,
                    self.left_wheel,
                    self.right_wheel,
                ]
            ],
            device=device,
        )
