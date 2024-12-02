# ./isaaclab.sh -p Test_MSP/10_Registering_Environment.py --task Isaac-Cartpole-v0 --num_vens 64
"""Launch Issac Sim first"""
import argparse
from omni.isaac.lab.app import AppLauncher
#Add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac lab env.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operation")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to create")
parser.add_argument("--task", type=str, default=None, help="Name of the task to load")
#Append Applaucher cli args
AppLauncher.add_app_launcher_args(parser)  
args_cli= parser.parse_args()
#Launch ominverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import omni.isaac.lab_tasks # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg

def main():
    """Random actions agent with issac lab environment"""
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    #print info
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # sample actions from -1 to 1
            actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            # apply actions
            env.step(actions)

    # close the simulator
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()