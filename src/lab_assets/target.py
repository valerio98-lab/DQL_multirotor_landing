import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

IW_HUB_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Idealworks/iw_hub.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            angular_damping=0.05,
            max_linear_velocity=float("inf"),
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=5729.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
            sleep_threshold=5e-7,
            stabilization_threshold=0.0009999999,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            stabilization_threshold=0.0009999999,
            sleep_threshold=5e-7,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            ".*": 0.0,
        },
    ),
    actuators={
        "left_wheel_joint": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel_joint"],
            velocity_limit=57295780.0,
            stiffness=0.0,
            damping=17453292.0,
            friction=0.0,
        ),
        "right_wheel_joint": ImplicitActuatorCfg(
            joint_names_expr=["right_wheel_joint"],
            velocity_limit=57295780.0,
            stiffness=0.0,
            damping=17453292.0,
            friction=0.0,
        ),
    },
)
