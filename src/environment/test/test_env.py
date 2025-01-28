# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to create a rigid object and interact with it.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/01_assets/run_rigid_object.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with a rigid object.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg, Articulation, DeformableObject, DeformableObjectCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab_assets import CARTPOLE_CFG


def design_scene():
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # Create separate groups called "Origin1", "Origin2", "Origin3"
    # Each group will have a robot in it
    coords = [
        [0.25, 0.25, 0.0],  # cone
        [-0.25, 0.25, 0.0],  # cone
        [0.25, -0.25, 0.0],  # cone
        [-0.25, -0.25, 0.0],  # cone
        [0.0, 0.0, -1.0],  # cartpole
        [-1.0, 0.0, -1.0],  # cartpole
        [0.0, 0.75, 0.0],
        [-0.50, 0.50, 0.0],
        [0.0, -0.55, 0.0],
        [-1, -0.5, 0.0],
    ]

    origins = {"cone": coords[:4], "cartpole": coords[4:6], "cube": coords[6:]}
    for k, v in origins.items():
        for idx, origin in enumerate(v):
            prim_utils.create_prim(f"/World/Origin_{k}{idx}", "Xform", translation=origin)

    ## Deformable object
    cube_cfg = DeformableObjectCfg(
        prim_path="/World/Origin_cube.*/Cube",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.2, 0.2, 0.2),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
            physics_material=sim_utils.DeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=1e5),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        debug_vis=True,
    )
    cube_object = DeformableObject(cfg=cube_cfg)

    # Articulation cartpole
    cartpole_cfg = CARTPOLE_CFG.copy()
    cartpole_cfg.prim_path = "/World/Origin_cartpole.*/Robot"
    cartpole = Articulation(cfg=cartpole_cfg)

    # Rigid Object
    cone_cfg = RigidObjectCfg(
        prim_path="/World/Origin_cone.*/Cone",
        spawn=sim_utils.ConeCfg(
            radius=0.1,
            height=0.2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    cone_object = RigidObject(cfg=cone_cfg)

    # return the scene information
    scene_entities = {"cone": cone_object, "cartpole": cartpole, "cube": cube_object}
    return scene_entities, origins


""" 
Root state è un tensore di dimensione (N, 13) dove N è il numero di oggetti da simulare mentre
13 è il numero di variabili per ogni oggetto.

Le prime 3 variabili definiscono la posizione dell'oggetto (x,y,z), le successive 4 il suo orientamento (quaternion). 
Quindi 7 variabili per la posizione e l'orientamento.

Le successive 3 variabili definiscono la velocità lineare dell'oggetto (vx, vy, vz) 
mentre le ultime 3 la velocità angolare (wx, wy, wz)
Quindi 6 variabili per la velocità lineare e angolare.

13 variabili in totale.

"""


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, RigidObject], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    cone_object = entities["cone"]
    cartpole = entities["cartpole"]
    cube_object = entities["cube"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    nodal_kinematic_target = cube_object.data.nodal_kinematic_target.clone()
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 250 == 0:
            # reset counters
            sim_time = 0.0
            count = 0

            # reset root state
            root_state = cone_object.data.default_root_state.clone()
            root_state_cartpole = cartpole.data.default_root_state.clone()
            nodal_state = cube_object.data.default_nodal_state_w.clone()

            pos_w = torch.rand(cube_object.num_instances, 3, device=sim.device) * 0.1 + origins[6:]
            quat_w = math_utils.random_orientation(cube_object.num_instances, device=sim.device)
            nodal_state[..., :3] = cube_object.transform_nodal_pos(nodal_state[..., :3], pos_w, quat_w)

            # sample a random position on a cylinder around the origins
            root_state[:, :3] += origins[:4]
            root_state[:, :3] += math_utils.sample_cylinder(
                radius=0.5, h_range=(0.25, 1.0), size=cone_object.num_instances, device=cone_object.device
            )
            # print("Root state shape: ", root_state.shape)
            # print("Root state shape: ", root_state_cartpole.shape)
            root_state_cartpole[:, :3] += origins[4:6]

            # write root state to simulation
            cone_object.write_root_pose_to_sim(root_state[:, :7])
            cone_object.write_root_velocity_to_sim(root_state[:, 7:])
            cartpole.write_root_pose_to_sim(root_state_cartpole[:, :7])
            cartpole.write_root_velocity_to_sim(root_state_cartpole[:, 7:])
            cube_object.write_nodal_state_to_sim(nodal_state)

            joint_pos, joint_vel = cartpole.data.default_joint_pos.clone(), cartpole.data.default_joint_vel.clone()
            joint_pos += torch.randn_like(joint_pos) * 0.1
            cartpole.write_joint_state_to_sim(joint_pos, joint_vel)

            # Write the nodal state to the kinematic target and free all vertices

            nodal_kinematic_target[..., :3] = nodal_state[..., :3]
            nodal_kinematic_target[..., 3] = 1.0
            cube_object.write_nodal_kinematic_target_to_sim(nodal_kinematic_target)

            # reset buffers
            cone_object.reset()
            cartpole.reset()
            cube_object.reset()
            print("----------------------------------------")
            print("[INFO]: Resetting object state...")

        # update the kinematic target for cubes at index 0 and 3

        # we slightly move the cube in the z-direction by picking the vertex at index 0

        nodal_kinematic_target[[0, 3], 0, 2] += 0.001
        # set vertex at index 0 to be kinematically constrained
        # 0: constrained, 1: free
        nodal_kinematic_target[[0, 3], 0, 3] = 0.0

        efforts = torch.randn_like(cartpole.data.joint_pos) * 8.0
        # -- apply action to the robot
        cartpole.set_joint_effort_target(efforts)

        # apply sim data
        cartpole.write_data_to_sim()
        cone_object.write_data_to_sim()
        cube_object.write_nodal_kinematic_target_to_sim(nodal_kinematic_target)
        cube_object.write_data_to_sim()

        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        cone_object.update(sim_dt)
        cartpole.update(sim_dt)
        cube_object.update(sim_dt)
        # print the root position
        if count % 50 == 0:
            print(f"Root position (in world): {cone_object.data.root_state_w[:, :3]}")
            print(f"Root position (in world): {cartpole.data.root_state_w[:, :3]}")
            print(f"Root position (in world): {cube_object.data.nodal_state_w[:, :3]}")


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[1.5, 0.0, 1.0], target=[0.0, 0.0, 0.0])
    # Design scene
    scene_entities, scene_origins = design_scene()
    scene_origins_cone = torch.tensor(scene_origins["cone"], device=sim.device)
    scene_origins_cartpole = torch.tensor(scene_origins["cartpole"], device=sim.device)
    scene_origins_cube = torch.tensor(scene_origins["cube"], device=sim.device)
    scene_origins = torch.cat((scene_origins_cone, scene_origins_cartpole, scene_origins_cube), dim=0)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
