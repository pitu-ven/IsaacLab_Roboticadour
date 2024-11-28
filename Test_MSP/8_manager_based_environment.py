# Floating cube environment with custom action term for PD control
#./isaaclab.sh -p source/standalone/tutorials/03_envs/create_cube_base_env.py --num_envs 32
# Quadrupedal locomotion environment with a policy that interacts with the environment
#./isaaclab.sh -p source/standalone/tutorials/03_envs/create_quadruped_base_env.py --num_envs 32

"""Launch Isaac Sim Simulator first."""
import argparse
from omni.isaac.lab.app import AppLauncher
# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on create a simple environment with a cartpole.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import math
import torch
import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleSceneCfg

@configclass
class ActionsCfg:
    """Action class for the environment."""
    joint_efforts = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=5.0)

@configclass
class ObservationsCfg:
    """Observation class for the environment"""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observation for policy group."""
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Configurations for the event."""
    add_pole_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["pole"]),
            "mass_distribution_params": (0.1, 0.5),
            "operation": "add",
        },
    )
    #on reset
    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.1, 0.1),
        },
    )
    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "position_range": (-0.125*math.pi,0.125*math.pi),
            "velocity_range": (-0.01*math.pi,0.01*math.pi),
        },
    )

@configclass
class CartpoleEnvCfg(ManagerBasedEnvCfg):
    """Configuration for cartpole environment."""
    scene = CartpoleSceneCfg(num_envs=1024, env_spacing=2.5)
    #Basic settings
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        self.viewer.eye = [4.5, 0.0, 6.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]
        #Step settings
        self.decimation = 4 #env step every 4 sim steps: 200Hz/4 = 50Hz
        self.sim.dt = 0.005 #sim step every 5ms -> 200Hz

def main():
    """Main function."""
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    #Setup base environments
    env = ManagerBasedEnv(cfg=env_cfg)

    #Simulate physics 
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            if count %3000 == 0:
                count = 0
                env.reset()
                print("-" *80)
                print("[INFO]: Resetting environment state...")
            #Sample random actions
            joint_efforts = torch.rand_like(env.action_manager.action)
            #Step environment
            obs, _ = env.step(joint_efforts)
            #Print current pole orientation
            print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            count += 1
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()