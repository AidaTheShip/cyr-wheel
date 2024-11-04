import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np

# Load your .mat file (replace 'your_file.mat' with the actual file path)
mat_data = scipy.io.loadmat('./data/Trial21 Kinematics.mat')

# Extract center of mass data
center_of_mass = mat_data['Center_of_Mass'][0, 0]  # Adjusted for nested structure

# Define joint positions to plot (example: left and right hip angles)
joints_to_plot = ['Left_Hip_Angles', 'Right_Hip_Angles']
joint_positions = {joint: mat_data[joint][0, 0] for joint in joints_to_plot}

# Setting up the 3D plot for animation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set plot limits based on center of mass for clarity
ax.set_xlim(np.min(center_of_mass[:, 0]), np.max(center_of_mass[:, 0]))
ax.set_ylim(np.min(center_of_mass[:, 1]), np.max(center_of_mass[:, 1]))
ax.set_zlim(np.min(center_of_mass[:, 2]), np.max(center_of_mass[:, 2]))

# Initial empty plots for center of mass and each joint
center_of_mass_line, = ax.plot([], [], [], 'o-', label="Center of Mass")
joint_lines = [ax.plot([], [], [], 'o-')[0] for _ in joint_positions]

# Initialization function
def init():
    center_of_mass_line.set_data([], [])
    center_of_mass_line.set_3d_properties([])
    for line in joint_lines:
        line.set_data([], [])
        line.set_3d_properties([])
    return [center_of_mass_line] + joint_lines

# Animation function
def animate(i):
    # Update center of mass line
    center_of_mass_line.set_data(center_of_mass[:i, 0], center_of_mass[:i, 1])
    center_of_mass_line.set_3d_properties(center_of_mass[:i, 2])
    
    # Update joint lines for each joint
    for j, (joint_name, joint_data) in enumerate(joint_positions.items()):
        joint_lines[j].set_data(joint_data[:i, 0], joint_data[:i, 1])
        joint_lines[j].set_3d_properties(joint_data[:i, 2])
        
    return [center_of_mass_line] + joint_lines

# Creating the animation
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(center_of_mass), interval=30, blit=True)
ani.save('CoM_Movement.mp4', writer='ffmpeg', fps=30)  # For .mp4


# Display the legend and show the animation
plt.tight_layout()
plt.legend()
plt.show()
