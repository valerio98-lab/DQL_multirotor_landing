"""
Quacopter environment.
"""

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    # https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/register_rl_env_gym.html#:~:text=The%20id%20argument,Anymal-C-v0.
    id="Isaac-Quadrotor-Landing-V0",
    entry_point=f"{__name__}.direct_env:QuadrotorEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:QuadrotorEnvCfg",
    },
)
