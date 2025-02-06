import torch
import gymnasium as gym
from omni.isaac.lab.utils.math import matrix_from_quat
from omni.isaac.lab.utils.math import quat_from_euler_xyz, quat_inv, quat_mul, euler_xyz_from_quat

"""
La classe non è utilizzata nel progetto, ma è stata implementata per mostrare come si può controllare il drone in Isaac Lab
La funzione di pre_physics_step realmente usata è chiamata direttamente dentro direct_env. Questa è tenuta qui come riferimento
"""


class DroneController:
    def __init__(self, agent, cfg):
        self._agent = agent  # Drone agent in Isaac Lab
        self.cfg = cfg  # Configurazione dei guadagni e parametri di controllo
        self._agent_weight = cfg.agent_weight  # Peso del drone

        # Guadagni del controllore (simili a PID)
        self.gain_attitude = torch.tensor([3.0, 3.0, 0.035], device=cfg.device)  # Guadagni per roll, pitch, yaw
        self.gain_angular_rate = torch.tensor([0.5, 0.5, 0.02], device=cfg.device)  # Guadagni per le velocità angolari

    def _pre_physics_step(
        self, actions: torch.Tensor, agent_orientation_quat: torch.Tensor, agent_angular_velocity: torch.Tensor
    ):
        self._actions = actions.clone()

        # 1. Calcolo della thrust (spinta) verticale
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._thrust[:, 0, 2] = self._actions[:, 0] * self._agent_weight

        # 2. Ottenere l'orientamento attuale del drone come quaternione
        current_quaternion = agent_orientation_quat

        # 3. Convertire le azioni RL in angoli desiderati (roll e pitch)
        desired_roll = self._actions[:, 1]
        desired_pitch = self._actions[:, 2]
        desired_yaw_rate = self._actions[:, 3]  # Controllo del yaw rate

        # 4. Creare il quaternione desiderato
        desired_quaternion = quat_from_euler_xyz(desired_roll, desired_pitch, torch.zeros_like(desired_roll))

        # 5. Calcolare l'errore angolare tra il quaternione attuale e quello desiderato
        quaternion_error = quat_mul(quat_inv(desired_quaternion), current_quaternion)

        # 6. Convertire l'errore angolare in angoli di Eulero (roll, pitch, yaw)
        angle_error = torch.stack(euler_xyz_from_quat(quaternion_error), dim=-1)

        # 7. Ottenere la velocità angolare attuale del drone
        current_angular_velocity = agent_angular_velocity

        # 8. Definire la velocità angolare desiderata (zero per roll e pitch, valore specifico per yaw)
        desired_angular_velocity = torch.zeros_like(current_angular_velocity)
        desired_angular_velocity[:, 2] = desired_yaw_rate

        # 9. Calcolare l'errore nella velocità angolare
        angular_velocity_error = current_angular_velocity - desired_angular_velocity

        # 10. Calcolare i momenti da applicare (controllore tipo PID)
        moment = -self.gain_attitude * angle_error - self.gain_angular_rate * angular_velocity_error

        # 11. Applicare i momenti calcolati
        self._moment = torch.zeros_like(self._actions[:, 1:4])
        self._moment[:, :] = moment

    def _apply_action(self):
        # Applicazione delle forze e dei momenti al drone
        self._agent.set_external_force_and_torque(
            self._thrust.unsqueeze(1), self._moment.unsqueeze(1), body_ids=self.cfg.body_id
        )


# def _pre_physics_step(self, actions: torch.Tensor):
#     self._actions = actions.clone().clamp(-1.0, 1.0)

#     # Convert normalized action to desired pitch angle
#     desired_pitch = self._actions[:, 0] * self.cfg.max_pitch_angle  # Scale appropriately

#     # Apply desired pitch angle as a position target
#     self.set_joint_position_target(desired_pitch, joint_ids=[self.cfg.pitch_joint_id])

#     # Apply thrust (same as before)
#     self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_wei
