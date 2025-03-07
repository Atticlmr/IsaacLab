# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to simulate a quadcopter.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/quadcopter_visual.py

"""

"""Launch Isaac Sim Simulator first."""

# ===================================================================
import matplotlib.pyplot as plt
import numpy as np



import argparse
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to simulate a quadcopter.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

# =============================================================================
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, RayCasterCameraCfg, TiledCameraCfg
from isaaclab.sensors.ray_caster import patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


##
# Pre-defined configs
##
# from isaaclab_assets import CRAZYFLIE_CFG  # isort:skip
from isaaclab_assets import CRAZYFLIE_VISUAL_CFG # isort:skip

# 这个是生成场景的配置，默认的崎岖地形场景，先做测试用，回头换成森林的
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort:skip




def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera

    # 这个是相机坐标eye和相机朝向target
    sim.set_camera_view(eye=[1.0, 0.5, 1.0], target=[0.0, 0.0, 0.5])

    # Spawn things into stage


    # # Ground-plane
    
    
    # cfg = sim_utils.GroundPlaneCfg()
    # 目前仅仅改了cfg名字增加了一个cfg，可能有别的方法加载地形usd文件
    cfg = sim_utils.forestPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    

    # Lights
    # 默认就是a平行光，适合模拟a太阳光
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    

    # Robots
    # 默认是crazyflie，目前只是改了名字3.5
    robot_cfg = CRAZYFLIE_VISUAL_CFG.replace(prim_path="/World/Crazyflie")
    robot_cfg.spawn.func("/World/Crazyflie", robot_cfg.spawn, translation=robot_cfg.init_state.pos)


    # create handles for the robots
    robot = Articulation(robot_cfg)

    # Play the simulator
    sim.reset()

    # Fetch relevant parameters to make the quadcopter hover in place
    prop_body_ids = robot.find_bodies("m.*_prop")[0]
    robot_mass = robot.root_physx_view.get_masses().sum()
    gravity = torch.tensor(sim.cfg.gravity, device=sim.device).norm()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Define simulation stepping
    # dt默认值好象是simulation_cfg.py的1/60,先不改
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    # Simulate physics 
    while simulation_app.is_running():
        # reset 
        if count % 2000 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset dof state
            joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])
            robot.write_root_velocity_to_sim(robot.data.default_root_state[:, 7:])
            robot.reset()
            # reset command
            print(">>>>>>>> Reset!")
        # apply action to the robot (make the robot float in place)
        forces = torch.zeros(robot.num_instances, 4, 3, device=sim.device)
        torques = torch.zeros_like(forces)
        forces[..., 2] = robot_mass * gravity / 4.0
        robot.set_external_force_and_torque(forces, torques, body_ids=prop_body_ids)
        robot.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        robot.update(sim_dt)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
