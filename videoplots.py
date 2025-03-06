import numpy as np
import scipy.io as sci
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pca import pca_svd
'''

Load stuff

'''

#For trial 1 for now
wheel_data = sci.loadmat(f'data/Contact_point/Trial0001.mat')
wheel_COM = wheel_data['Trial0001'][0][0][5][0][0]['Positions'][0]
print(np.shape(wheel_COM))
wheel_keys = list(wheel_data.keys())
#rotations
rot_data = wheel_data[wheel_keys[-1]] #'Trial0001' key
rotations = rot_data[0][0][-1][0][0]['Rotations'][0]
wheel_COM = rot_data[0][0][-1][0]['Positions'][0][0].T / 1000



def extract_rotation_matrices(rotation_array):
    t = rotation_array.shape[1]  # Number of time steps
    rotation_matrices = [rotation_array[:, i].reshape(3, 3) for i in range(t)]
    return rotation_matrices
rot_mat = np.array(extract_rotation_matrices(rotations))
rot_mat = np.transpose(rot_mat, axes=(0, 2, 1))

# wheel_COM = wheel_data['Trial0001'][0][0][5][0][0]['Positions'][0] #(3,t)

segment_data = sci.loadmat('./data/Kinematics + EMG/Trial 1 Segment CoM.mat') #body segments - arm, leg, ect.

# axis = 0 #0 for x, 1, for y, 2 for z
axis_labels = ['horizontal in wheel plane','perpendicular to wheel plane','vertical in wheel plane']
# Printing the keys in our .mat file. 
# print(mat_data.keys())  
keys = list(segment_data.keys())[6:]



time_passed = len(segment_data[keys[-1]][0][0])
time = range(time_passed)

# Wheel center of mass coordinates
def COM_coordinates(v, rot, com):
    array = []
    for t in range(np.shape(v)[0]):
        
        array.append(np.dot(rot[t].T,  v[t]- com[t]))
    return np.array(array)

LeftUpperArmCoM = segment_data['LeftUpperArmCoM'][0][0]
LeftForearmCoM = segment_data['LeftForearmCoM'][0][0]
LeftHandCoM = segment_data['LeftHandCoM'][0][0]
RightUpperArmCoM = segment_data['RightUpperArmCoM'][0][0]
RightForearmCoM = segment_data['RightForearmCoM'][0][0]
RightHandCoM = segment_data['RightHandCoM'][0][0]
LeftThighCoM = segment_data['LeftThighCoM'][0][0]
LeftShankCoM = segment_data['LeftShankCoM'][0][0]
LeftFootCoM = segment_data['LeftFootCoM'][0][0]
RightThighCoM = segment_data['RightThighCoM'][0][0]
RightShankCoM = segment_data['RightShankCoM'][0][0]
RightFootCoM = segment_data['RightFootCoM'][0][0]

LUA = COM_coordinates(LeftUpperArmCoM, rot_mat, wheel_COM)
LFA = COM_coordinates(LeftForearmCoM, rot_mat, wheel_COM)
LH = COM_coordinates(LeftHandCoM, rot_mat, wheel_COM)
RUA = COM_coordinates(RightUpperArmCoM, rot_mat, wheel_COM)
RFA = COM_coordinates(RightForearmCoM, rot_mat, wheel_COM)
RH = COM_coordinates(RightHandCoM, rot_mat, wheel_COM)
LT = COM_coordinates(LeftThighCoM, rot_mat, wheel_COM)
LS = COM_coordinates(LeftShankCoM, rot_mat, wheel_COM)
LF = COM_coordinates(LeftFootCoM, rot_mat, wheel_COM)
RT = COM_coordinates(RightThighCoM, rot_mat, wheel_COM)
RS = COM_coordinates(RightShankCoM, rot_mat, wheel_COM)
RF = COM_coordinates(RightFootCoM, rot_mat, wheel_COM)

COM_segs = [LH, LFA, LUA, RH, RFA, RUA, LT, LS, LF, RT, RS,RF]

'''

Plot all the segments and the wheel in 3D

'''

# Define the vector components
  # Origin point
# Vector components (x, y, z)
xvec = np.array([1, 0, 0])  
yvec = np.array([0, 1, 0])
zvec = np.array([0, 0, 1])

# Generate points for the ring centered at (0, 0, 0)
theta = np.linspace(0, 2 * np.pi, 100)  # Angle range for the ring
ring_radius = 1.0                       # Radius of the ring

# Combine right_vector and face_vector to generate the ring
ring_points = ring_radius * (
    np.outer(np.cos(theta), xvec) +
    np.outer(np.sin(theta), zvec)
)





# for t in range(0,np.shape(rot_mat)[0],4):
#     origin = [0,0,0]


#     # Create a 3D plot
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # Plot the ring centered at (0, 0, 0)
#     ax.plot(ring_points[:, 0], ring_points[:, 1], ring_points[:, 2], label="Ring", color='black')
#     ax.scatter(*origin, color = "black")
#     # # Plot the basis vectors
#     # ax.quiver(*origin, *xvec,  length=np.linalg.norm(xvec), normalize=False, color = "black")
#     # ax.quiver(*origin, *yvec, length=np.linalg.norm(yvec), normalize=False, color = "black")
#     # ax.quiver(*origin, *zvec, length=np.linalg.norm(zvec), normalize=False, color = "black")


#     ax.scatter(*LUA[t], color ='b')
#     ax.scatter(*RUA[t], color ='b')
#     ax.scatter(*RFA[t], color ='b')
#     ax.scatter(*LFA[t], color ='b')
#     ax.scatter(*LH[t], color ='b')
#     ax.scatter(*RH[t], color ='b')
#     ax.scatter(*LT[t], color ='b')
#     ax.scatter(*RS[t], color ='b')
#     ax.scatter(*RT[t], color ='b')
#     ax.scatter(*LS[t], color ='b')
#     ax.scatter(*LF[t], color ='b')
#     ax.scatter(*RF[t], color ='b')
 
    


#     # Set labels and limits for clarity
#     ax.set_xlim([-2, 2])
#     ax.set_ylim([-2, 2])
#     ax.set_zlim([-2, 2])
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')

#     # Show the plot
#     plt.savefig(f"pca/Trial1/all_segments/frame{t}.png")
#     plt.close(fig)
#     plt.clf()


'''

Plot the project onto the y-z plane (to see motion in and out of wheel)

'''
# z_range = np.linspace(-ring_radius, ring_radius, 50)
# y_range = np.zeros(len(z_range))

# for t in range(0,np.shape(rot_mat)[0],4):
#     plt.clf()
#     plt.plot(y_range, z_range)



#     plt.scatter(LUA[t][1], LUA[t][2], color='b')
#     plt.scatter(RUA[t][1], RUA[t][2], color='b')
#     plt.scatter(RFA[t][1], RFA[t][2], color='b')
#     plt.scatter(LFA[t][1], LFA[t][2], color='b')
#     plt.scatter(LH[t][1], LH[t][2], color='b')
#     plt.scatter(RH[t][1], RH[t][2], color='b')
#     plt.scatter(LT[t][1], LT[t][2], color='b')
#     plt.scatter(RS[t][1], RS[t][2], color='b')
#     plt.scatter(RT[t][1], RT[t][2], color='b')
#     plt.scatter(LS[t][1], LS[t][2], color='b')
#     plt.scatter(LF[t][1], LF[t][2], color='b')
#     plt.scatter(RF[t][1], RF[t][2], color='b')

#     plt.xlim(-2,2)
#     plt.ylim(-2,2)

#     plt.savefig(f"pca/Trial1/all_segs2D/frame{t}.png")
    



''' Plot 3D with Torso and Head '''

# #estimate torso position as average between upper arms and thighs
# torso_seg = (LUA+RUA+LT+RT) / 4
# #estimate head position as torso plus 50 cm
# head_seg = np.copy(torso_seg)
# head_seg[:,2]+=0.5



# xvec = np.array([1, 0, 0])  
# yvec = np.array([0, 1, 0])
# zvec = np.array([0, 0, 1])

# # Generate points for the ring centered at (0, 0, 0)
# theta = np.linspace(0, 2 * np.pi, 100)  # Angle range for the ring
# ring_radius = 1.0                       # Radius of the ring

# # Combine right_vector and face_vector to generate the ring
# ring_points = ring_radius * (
#     np.outer(np.cos(theta), xvec) +
#     np.outer(np.sin(theta), zvec)
# )

# origin = [0,0,0]
# for t in range(0,np.shape(rot_mat)[0],8): #Divide total frames by 8, then render at 25 fps
    


#     # Create a 3D plot
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # Plot the ring centered at (0, 0, 0)
#     ax.plot(ring_points[:, 0], ring_points[:, 1], ring_points[:, 2], label="Ring", color='black')
#     ax.scatter(*origin, color = "black")
#     # # Plot the basis vectors
#     # ax.quiver(*origin, *xvec,  length=np.linalg.norm(xvec), normalize=False, color = "black")
#     # ax.quiver(*origin, *yvec, length=np.linalg.norm(yvec), normalize=False, color = "black")
#     # ax.quiver(*origin, *zvec, length=np.linalg.norm(zvec), normalize=False, color = "black")


#     ax.scatter(*LUA[t], color ='b')
#     ax.scatter(*RUA[t], color ='b')
#     ax.scatter(*RFA[t], color ='b')
#     ax.scatter(*LFA[t], color ='b')
#     ax.scatter(*LH[t], color ='b')
#     ax.scatter(*RH[t], color ='b')
#     ax.scatter(*LT[t], color ='b')
#     ax.scatter(*RS[t], color ='b')
#     ax.scatter(*RT[t], color ='b')
#     ax.scatter(*LS[t], color ='b')
#     ax.scatter(*LF[t], color ='b')
#     ax.scatter(*RF[t], color ='b')
#     ax.scatter(*head_seg[t], color = "red")
#     ax.scatter(*torso_seg[t], color = "red")
    


#     # Set labels and limits for clarity
#     ax.set_xlim([-1.5, 1.5])
#     ax.set_ylim([-1, 1])
#     ax.set_zlim([-1.5, 1.5])
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_box_aspect([1, 1, 1])

#     # ax.view_init(elev=0, azim=90) #set camera

#     # Show the plot
#     plt.savefig(f"pca/Trial1/all_segs_head_torso/FramesBackRight/frame{t}.png")
#     plt.close(fig)
#     plt.clf()



''' Center of mass estimation '''



# Center of mass estimation ###


body_segment_masses = np.array([ #Body mass estimates in kg from chat GPT 'These masses are for a 45-year-old woman with a mass of 70 kg and approximately 30% body fat'
    0.4545,  # LeftHand
    1.2727,  # LeftForearm
    2.0909,  # LeftUpperArm
    0.4545,  # RightHand
    1.2727,  # RightForearm
    2.0909,  # RightUpperArm
    10.9091, # LeftThigh
    3.6364,  # LeftShank
    1.2727,  # LeftFoot
    10.9091, # RightThigh
    3.6364,  # RightShank
    1.2727   # RightFoot
])


#estimate torso position as average between upper arms and thighs
torso_seg = (LUA+RUA+LT+RT) / 4
#estimate head position as torso plus 50 cm
head_seg = np.copy(torso_seg)
head_seg[:,2]+=0.5

remaining_mass = 70 - np.sum(body_segment_masses)
head_mass = remaining_mass / 5
torso_mass = remaining_mass - head_mass
body_segment_masses = np.append(body_segment_masses, [torso_mass, head_mass], axis=None)
COM_segs.append(torso_seg)
COM_segs.append(head_seg)
COM = np.zeros(shape = np.shape(COM_segs[0][1000:4400]))
for i in range(len(body_segment_masses)):
    COM += body_segment_masses[i]*COM_segs[i][1000:4400]
COM = COM / 70 #normalize by 70kg

# Optional: Plot COM in 2D
for i in range(3):
    plt.plot(np.linspace(1000/200, 4400/200, len(COM.T[i][1000:4400])), COM.T[i][1000:4400], label = ['x','y','z'][i])
plt.legend()
plt.title("Components of COM")
plt.xlabel("time")
plt.ylabel("Distance")
plt.savefig("Trial1_analysis/COM/components")
plt.show()
plt.clf()

fig, ax = plt.subplots(3,1)
fs = 200
for i in range(3):
    signal = COM.T[i] - np.mean(COM.T[i])
    fft_result = np.fft.fft(signal)  # FFT of the signal
    frequencies = np.fft.fftfreq(len(signal), 1 / fs)  # Corresponding frequencies

    # Compute the magnitude spectrum
    magnitude = np.abs(fft_result)
    max_index= np.argmax(magnitude[:fs // 2])
    max_freq = frequencies[max_index]
    print(max_freq)
    ax[i].plot(frequencies[:fs // 2], magnitude[:fs // 2])
    ax[i].set_title(['x','y','z'][i])
    ax[i].set_xlabel("Frequency")
    ax[i].set_ylabel("Amplitude")
    
# Compute the magnitude spectrum

plt.savefig("Trial1_analysis/COM/fft")
xvec = np.array([1, 0, 0])  
yvec = np.array([0, 1, 0])
zvec = np.array([0, 0, 1])


''' Plot 3D'''
# # Generate points for the ring centered at (0, 0, 0)
# theta = np.linspace(0, 2 * np.pi, 100)  # Angle range for the ring
# ring_radius = 1.0                       # Radius of the ring

# # Combine right_vector and face_vector to generate the ring
# ring_points = ring_radius * (
#     np.outer(np.cos(theta), xvec) +
#     np.outer(np.sin(theta), zvec)
# )

# origin = [0,0,0]
# for t in range(0,np.shape(rot_mat)[0],8):
    
#     # Create a 3D plot
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # Plot the ring centered at (0, 0, 0)
#     ax.plot(ring_points[:, 0], ring_points[:, 1], ring_points[:, 2], label="Ring", color='black')
#     ax.scatter(*origin, color = "black")

#     ax.scatter(*COM[t], color = "red")

#     ax.set_xlim([-1.5, 1.5])
#     ax.set_ylim([-1, 1])
#     ax.set_zlim([-1.5, 1.5])

#     ax.set_box_aspect([1, 1, 1])

#     # ax.view_init(elev=0, azim=90) #set camera. 90 azimuth is Front. Default is right back
#     plt.savefig(f"pca/Trial1/COM/BackRight/frame{t}.png")
#     plt.close()

''' 

Reduced data (from PCA)
parameter k decides number of principal components to include (in order of explained variance)

'''

num_segs = 12
data_matrix = []

for i in range(num_segs):
    
        
        data = segment_data[keys[i]][0][0]
        segment_COM = COM_coordinates(data, rot_mat, wheel_COM).T
        data_matrix.append(segment_COM[0][1000:4400]) #Remove getting on and getting off
        data_matrix.append(segment_COM[1][1000:4400])
        data_matrix.append(segment_COM[2][1000:4400])

        #Add EMG data

print("Shape of data matrix " + str(np.shape(data_matrix)))



data_mat = np.array(data_matrix)

k = 1 #Which component (python index)

projected, explained_variance, Vt, reduced = pca_svd(data_mat, k = k)

# # Estimate Head and Torso (optional)
# # estimate torso position as average between upper arms and thighs
# torso_seg = (LUA+RUA+LT+RT) / 4
# # estimate head position as torso plus 50 cm
# head_seg = np.copy(torso_seg)
# head_seg[:,2]+=0.5



xvec = np.array([1, 0, 0])  
yvec = np.array([0, 1, 0])
zvec = np.array([0, 0, 1])

# Generate points for the ring centered at (0, 0, 0)
theta = np.linspace(0, 2 * np.pi, 100)  # Angle range for the ring
ring_radius = 1.0                       # Radius of the ring

# Combine right_vector and face_vector to generate the ring
ring_points = ring_radius * (
    np.outer(np.cos(theta), xvec) +
    np.outer(np.sin(theta), zvec)
)

origin = [0,0,0]


for t in range(0,3400,8): #Divide total frames by 8, then render at 25 fps
    


    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the ring centered at (0, 0, 0)
    ax.plot(ring_points[:, 0], ring_points[:, 1], ring_points[:, 2], label="Ring", color='black')
    ax.scatter(*origin, color = "black")
    # # Plot the basis vectors
    # ax.quiver(*origin, *xvec,  length=np.linalg.norm(xvec), normalize=False, color = "black")
    # ax.quiver(*origin, *yvec, length=np.linalg.norm(yvec), normalize=False, color = "black")
    # ax.quiver(*origin, *zvec, length=np.linalg.norm(zvec), normalize=False, color = "black")

    
    for i in range(0,np.shape(reduced)[0],3):
        ax.scatter(*[reduced[i][t],reduced[i+1][t],reduced[i+2][t]], color ='b')
    

    # Set labels and limits for clarity
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1.5, 1.5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])
    ax.set_title(f"PC {k+1}")

    # ax.view_init(elev=0, azim=90) #set camera

    # Show the plot
    plt.savefig(f"Trial1_analysis/PCA/Reduced/{k+1}/frame{t}.png")
    plt.close(fig)
    plt.clf()


