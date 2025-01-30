# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import omni.isaac.lab.sim as sim_utils
import torch
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.envs.ui import BaseEnvWindow
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import RayCaster, RayCasterCfg, patterns
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
from lab_assets.agent import CRAZYFLIE_CFG  # isort: skip
from omni.isaac.lab.markers import CUBOID_MARKER_CFG  # isort: skip
from lab_assets.target import IW_HUB_CFG  # isort: skip

XMAX, YMAX, ZMAX = 9.0, 9.0, 5.0


class QuadrotorEnvWindow(BaseEnvWindow):
    """Window manager for the Quadrotor environment."""

    def __init__(self, env: QuadrotorEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)  # type: ignore
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class QuadrotorEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 800.0
    decimation = 2
    action_space = 6
    observation_space = 12
    state_space = 0
    debug_vis = True

    ui_window_class_type = QuadrotorEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # agent
    agent: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Agent")  # type: ignore

    thrust_to_weight = 1.0  # 1.9
    moment_scale = 0.01

    ## target
    target: ArticulationCfg = IW_HUB_CFG.replace(prim_path="/World/envs/env_.*/Target")  # type: ignore

    height: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Agent/body",
        update_period=0.02,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0


@dataclass
class Observations:
    relative_position: torch.Tensor
    relative_velocity: torch.Tensor
    relative_acceleration: torch.Tensor
    relative_orientation: torch.Tensor

    def __repr__(self) -> str:
        return str(self.__dict__)


class QuadrotorEnv(DirectRLEnv):
    cfg: QuadrotorEnvCfg

    def __init__(self, cfg: QuadrotorEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        ## Fly zone
        self._fly_zone = torch.tensor([XMAX, YMAX, ZMAX], device=self.device)

        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
            ]
        }
        # Get specifics of the agent
        self._body_id = self._agent.find_bodies("body")[0]
        self._agent_masses = self._agent.root_physx_view.get_masses()[0]
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._agent_weight = (self._agent_masses.sum() * self._gravity_magnitude).item()

        # Get specifics of the target
        self._joint_id = self._target.find_joints(".*wheel_joint")[0]
        self._target_masses = self._target.root_physx_view.get_masses()[0]

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._agent = Articulation(self.cfg.agent)
        self._target = Articulation(self.cfg.target)
        self._sensor = RayCaster(self.cfg.height)
        self.scene.articulations["agent"] = self._agent
        self.scene.articulations["target"] = self._target
        self.scene.sensors["height"] = self._sensor

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    #############################VENDITTELLI

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self._actions[:, 0] * self._agent_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:4]
        self._target_action = self._actions[:, 4:6] * self._target_masses.sum()

    def _apply_action(self):
        self._agent.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)
        self._target.set_joint_velocity_target(self._target_action, joint_ids=self._joint_id)

    ###############################VENDITTELLI

    def _com_acc(self, robot_acc: torch.Tensor, robot_masses: torch.Tensor) -> torch.Tensor:
        """
        Compute CoM acceleration as the mean of the body accelerations
        """

        weighted_acc = robot_acc * robot_masses
        com_acc = torch.sum(weighted_acc, dim=1) / robot_masses.sum(dim=1)
        return com_acc

    def _get_observations(self) -> dict:
        agent_position = self._agent.data.root_state_w[:, :3]
        agent_orientation = self._agent.data.root_state_w[:, 3:7]
        agent_velocity = self._agent.data.root_state_w[:, 7:10]
        agent_com_acc = self._com_acc(
            self._agent.data.body_acc_w[:, :, :3], self._agent_masses.view(1, -1, 1).to("cuda")
        )

        target_position = self._target.data.root_state_w[:, :3]
        target_orientation = self._target.data.root_state_w[:, 3:7]
        target_velocity = self._target.data.root_state_w[:, 7:10]
        target_com_acc = self._com_acc(
            self._target.data.body_acc_w[:, :, :3], self._target_masses.view(1, -1, 1).to("cuda")
        )

        self.relative_pos_s, self.relative_orientation_s = subtract_frame_transforms(
            agent_position, agent_orientation, target_position, target_orientation
        )
        self.relative_vel_s = target_velocity - agent_velocity

        self.relative_acc_s = target_com_acc - agent_com_acc

        observation = Observations(
            relative_position=self.relative_pos_s,
            relative_velocity=self.relative_vel_s,
            relative_acceleration=self.relative_acc_s,
            relative_orientation=self.relative_orientation_s,
        )
        observation = {
            "observation": observation,
        }
        return observation

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._agent.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._agent.data.root_ang_vel_b), dim=1)
        # distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._agent.data.root_pos_w, dim=1)
        # distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            # "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1  ##troncamento
        # print(self._agent.data.root_pos_w[:, 2])
        agent_position = self._agent.data.root_pos_w[:, :3]
        died = torch.any(
            torch.logical_or(agent_position > self._fly_zone, agent_position < -self._fly_zone)
        )  ##terminazione
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._agent._ALL_INDICES

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._agent.data.root_pos_w[env_ids], dim=1
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        self._agent.reset(env_ids)  # type: ignore
        super()._reset_idx(env_ids)  # type: ignore
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        # Sample new commands
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)
        # Reset robot state
        joint_pos = self._agent.data.default_joint_pos[env_ids]
        joint_vel = self._agent.data.default_joint_vel[env_ids]
        default_root_state = self._agent.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._agent.write_root_pose_to_sim(default_root_state[:, :7], env_ids)  # type: ignore
        self._agent.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)  # type: ignore
        self._agent.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)  # type: ignore

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()  # type: ignore
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
