"""
This file deals with the cyrwheel model we will use for our reinforcement learning algorithm.

"""
import mujoco_py
import numpy as np
from modules import get_casadi_EOM

# Load the MuJoCo model
model = mujoco_py.load_model_from_path('path_to_your_xml_model.xml')
sim = mujoco_py.MjSim(model)

# Initialize CasADi functions (adapted from your code)
casadi_EOM, x, u = get_casadi_EOM(mass_of_the_wheel)

# Simulation parameters
dt = 0.01  # Simulation time step
n_steps = 1000  # Number of simulation steps

# Simulation loop
for step in range(n_steps):
    # Get current state from the simulation
    current_state = np.array([get_state_from_mujoco(sim)])
    
    # Calculate the new state using CasADi EOM
    # Note: You will need to adapt this part to your needs, including numerical integration
    new_state = casadi_EOM(current_state)
    
    # Apply the new state or forces/torques to the MuJoCo simulation
    # This step depends on how your EOM are defined and how they relate to the MuJoCo model
    apply_new_state_to_mujoco(sim, new_state)
    
    # Step the simulation forward
    sim.step()
    
    # Optional: render the simulation or log data
    sim.render()

