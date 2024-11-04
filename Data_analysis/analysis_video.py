import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# Load your .mat file (replace 'your_file.mat' with the actual file path)
mat_data = scipy.io.loadmat('./data/Trial21 Kinematics.mat')
center_of_mass_data = mat_data['Center_of_Mass'][0, 0]  # Adjust for nested structure if needed

# Define Kalman Filter parameters
dt = 1.0  # Time step (adjust if you know the frame rate)
n = len(center_of_mass_data)  # Number of time steps

# Initialize state [x, y, z, vx, vy, vz] where v is velocity
state = np.zeros((6, 1))  # Initial state
state[:3, 0] = center_of_mass_data[0]  # Initial position

# State Transition Matrix (A)
A = np.eye(6)
A[0, 3] = A[1, 4] = A[2, 5] = dt  # Incorporate time step for velocity

# Observation Matrix (H): we observe positions [x, y, z]
H = np.zeros((3, 6))
H[0, 0] = H[1, 1] = H[2, 2] = 1

# Covariances
Q = np.eye(6) * 0.1  # Process noise covariance (tweak for smoother results)
R = np.eye(3) * 0.5  # Measurement noise covariance (tweak for smoother results)
P = np.eye(6)  # Initial estimate error covariance

# Lists to store filtered values
filtered_positions = []

# Kalman filter loop over each time step
for measurement in center_of_mass_data:
    # Prediction step
    state = A @ state
    P = A @ P @ A.T + Q

    # Update step
    Z = measurement.reshape(3, 1)  # Measurement (CoM position at this time step)
    y = Z - H @ state  # Measurement residual
    S = H @ P @ H.T + R  # Residual covariance
    K = P @ H.T @ np.linalg.inv(S)  # Kalman gain
    state = state + K @ y
    P = (np.eye(6) - K @ H) @ P  # Update estimate error covariance

    # Store the filtered position
    filtered_positions.append(state[:3].flatten())

# Convert filtered positions to numpy array for plotting
filtered_positions = np.array(filtered_positions)

# Plot original vs. filtered CoM positions for each axis (X, Y, Z)
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(center_of_mass_data[:, 0], label="Original X Position")
plt.plot(filtered_positions[:, 0], label="Filtered X Position", linestyle='--')
plt.legend()
plt.title("Kalman Filtered Center of Mass Position - X")

plt.subplot(3, 1, 2)
plt.plot(center_of_mass_data[:, 1], label="Original Y Position")
plt.plot(filtered_positions[:, 1], label="Filtered Y Position", linestyle='--')
plt.legend()
plt.title("Kalman Filtered Center of Mass Position - Y")

plt.subplot(3, 1, 3)
plt.plot(center_of_mass_data[:, 2], label="Original Z Position")
plt.plot(filtered_positions[:, 2], label="Filtered Z Position", linestyle='--')
plt.legend()
plt.title("Kalman Filtered Center of Mass Position - Z")

plt.tight_layout()
plt.show()
