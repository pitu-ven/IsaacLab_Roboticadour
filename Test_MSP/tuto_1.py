import argparse #For headless mode etc...

from omni.isaac.lab.app import AppLauncher

#Create Parser
parser = argparse.ArgumentParser(description="Tutorial 1 -> Creating empty stage")
#Append appLauncher arguments
AppLauncher.add_app_launcher_args(parser)
#Parse arguments
args_cli = parser.parse_args()
#Launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


#Importing modules after sim is running
from omni.isaac.lab.sim import SimulationCfg, SimulationContext

#Initialize sim context
sim_config = SimulationCfg(dt=0.01)
sim = SimulationContext(sim_config)
#Set main camera
sim.set_camera_view([2.5,2.5,2.5],[0.0,0.0,0.0])

#Always reset the simulation before stepping it, for handles and other stuff
sim.reset()
#Now ready
print("[INFO]: Setup Complete...")

#Simulate physics
while simulation_app.is_running():
    #Step update
    sim.step()

#Exit
print("[INFO]: Exiting...")
simulation_app.close()