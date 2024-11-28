"""Launch Isaac Sim Simulator first."""
import argparse
from omni.isaac.lab.app import AppLauncher
parser = argparse.ArgumentParser(description="Interact with articulation")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import torch
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.sim import SimulationContext

#Import pre-defined config
from omni.isaac.lab_assets import CARTPOLE_CFG  # isort:skip

def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene"""
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    #Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    #Create separate groups, each have a robot inside
    origins = [[0.0,0.0,0.0],[-1.0,0.0,0.0]]
    #Origin 1
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    #Origin 2
    prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])

    #Articulation
    cartpole_cfg = CARTPOLE_CFG.copy()
    cartpole_cfg.prim_path = "/World/Origin.*/Robot"
    cartpole = Articulation(cfg=cartpole_cfg)

    #Return scene info
    scene_entities = {"cartpole": cartpole}
    return scene_entities, origins

def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    robot = entities["cartpole"]
    #Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    #Simulation loop
    while simulation_app.is_running():
        if count % 500 == 0:
            count = 0
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += origins
            robot.write_root_state_to_sim(root_state)
            #Set joint positions with noise
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            #Clear internal buffers
            robot.reset()
            print("[INFO]: Resetting robot state...")

        #Generate random joint efforts
        efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        #Apply actions to robot
        robot.set_joint_effort_target(efforts)
        #Write data to sim
        robot.write_data_to_sim()
        sim.step()
        count += 1
        #Buffers update
        robot.update(sim_dt)

def main():
    """Main function."""
    #Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    #Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    #Desing scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    #Play simulator
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene_entities, scene_origins)

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()