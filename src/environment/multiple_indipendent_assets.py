import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import math

import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Pre-defined configs
##
from omni.isaac.lab_assets import CARTPOLE_CFG  # isort:skip

from omni.isaac.lab_tasks.manager_based.classic.humanoid.humanoid_env_cfg import HumanoidEnvCfg


@configclass
class CartpoleScene(InteractiveSceneCfg):

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)))

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    robot1: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Cartpole1")
    robot2: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Cartpole2")


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(asset_name="robot1", joint_names=["slider_to_cart"], scale=5.0)
    joint_effort1 = mdp.JointEffortActionCfg(asset_name="robot2", joint_names=["slider_to_cart"], scale=25.0)


@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # on startup
    add_pole_mass1 = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(name="robot1", body_names=["pole"]),
            "mass_distribution_params": (0.5, 0.5),
            "operation": "add",
        },
    )

    add_pole_mass2 = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(name="robot2", body_names=["pole"]),
            "mass_distribution_params": (0.1, 0.5),
            "operation": "add",
        },
    )

    # on reset
    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name="robot1", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.1, 0.1),
        },
    )

    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name="robot1", joint_names=["cart_to_pole"]),
            "position_range": (-0.125 * math.pi, 0.125 * math.pi),
            "velocity_range": (-0.01 * math.pi, 0.01 * math.pi),
        },
    )


@configclass
class CartpoleEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the cartpole environment."""

    # Scene settings
    scene = CartpoleScene(num_envs=30, env_spacing=2.5)
    # Basic settings
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = [4.5, 0.0, 6.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]
        # step settings
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        self.sim.dt = 0.005  # sim step every 5ms: 200Hz


def main():
    """Main function."""
    # parse the arguments
    # print(f"{ISAACLAB_NUCLEUS_DIR}/Robots/Classic/Cartpole/cartpole.usd")
    # exit()
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup base environment
    env = ManagerBasedEnv(cfg=env_cfg)

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # sample random actions
            joint_efforts = torch.randn_like(env.action_manager.action)
            # step the environment
            obs, _ = env.step(joint_efforts)
            # print current orientation of pole
            print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
