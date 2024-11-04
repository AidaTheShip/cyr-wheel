import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Load your .mat file (replace 'your_file.mat' with the actual file path)
mat_data = scipy.io.loadmat('./data/Trial22 Kinematics.mat')
center_of_mass = mat_data['Center_of_Mass'][0, 0]  # Adjust for nested structure if necessary

# Set up 3D figure for animation
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("Forward motion (X)")
ax.set_ylabel("Vertical motion (Y)")
ax.set_zlabel("Depth (Z)")
ax.set_title("3D Rolling Wheel Animation Based on Center of Mass Data")

# Set plot limits based on CoM data
x_min, x_max = np.min(center_of_mass[:, 0]), np.max(center_of_mass[:, 0])
y_min, y_max = np.min(center_of_mass[:, 1]), np.max(center_of_mass[:, 1])
z_min, z_max = np.min(center_of_mass[:, 2]), np.max(center_of_mass[:, 2])
ax.set_xlim(x_min - 0.5, x_max + 0.5)
ax.set_ylim(y_min - 0.5, y_max + 0.5)
ax.set_zlim(z_min - 0.5, z_max + 0.5)

# Wheel parameters
wheel_radius = 0.1  # Adjust radius for visual clarity
num_steps = len(center_of_mass)

# Initialize elements for animation
wheel_line, = ax.plot([], [], [], 'o-', color="black", markersize=5, alpha=0.7)  # Center point of wheel
arrow_line, = ax.plot([], [], [], color="black", lw=2)  # Arrow showing direction on wheel circumference

# Initialize function for animation
def init():
    wheel_line.set_data([], [])
    wheel_line.set_3d_properties([])
    arrow_line.set_data([], [])
    arrow_line.set_3d_properties([])
    return wheel_line, arrow_line

# Animation function
def animate(i):
    # Update wheel center position
    x, y, z = center_of_mass[i, 0], center_of_mass[i, 1], center_of_mass[i, 2]
    wheel_line.set_data([x], [y])
    wheel_line.set_3d_properties([z])

    # Simulate rolling by rotating arrow on wheel
    angle = (i / num_steps) * 2 * np.pi  # Rotation angle based on time step
    arrow_x = x + wheel_radius * np.cos(angle)
    arrow_y = y + wheel_radius * np.sin(angle)
    arrow_z = z  # Keeping it on the same Z plane as the wheel center

    # Update arrow direction
    arrow_line.set_data([x, arrow_x], [y, arrow_y])
    arrow_line.set_3d_properties([z, arrow_z])

    return wheel_line, arrow_line

# Create the animation
ani = FuncAnimation(fig, animate, frames=range(0, num_steps, 10), init_func=init, blit=True)

# Save the animation as an .mp4 video
ani.save('3D_Rolling_Wheel_Animation.mp4', writer='ffmpeg', fps=30)

# If you want a .gif instead, use this line instead:
# ani.save('3D_Rolling_Wheel_Animation.gif', writer='imagemagick', fps=30)

plt.show()
