import numpy as np
import scipy.io as sci
import matplotlib.pyplot as plt
import scipy.spatial.transform as scirot


#For trial 1 for now
trial = '19'
wheel_data = sci.loadmat(f'data/Contact_Point/Trial00{trial}.mat')
wheel_keys = list(wheel_data.keys())
wheel_COM = wheel_data[wheel_keys[-1]][0][0][5][0][0]['Positions'][0]
print(np.shape(wheel_COM))
wheel_keys = list(wheel_data.keys())
#rotations
rot_data = wheel_data[wheel_keys[-1]] #'Trial0001' key
rotations = rot_data[0][0][-1][0][0]['Rotations'][0]
wheel_COM = rot_data[0][0][-1][0]['Positions'][0][0].T / 1000

raddeg = 360 / (2*np.pi)
start_frame = 1000
end_frame = 4400
fps = 200


def extract_rotation_matrices(rotation_array):
    t = rotation_array.shape[1]  # Number of time steps
    rotation_matrices = [rotation_array[:, i].reshape(3, 3) for i in range(t)]
    return rotation_matrices
rot_mat = np.array(extract_rotation_matrices(rotations))
# rot_mat = np.transpose(rot_mat, axes=(0, 2, 1))

# wheel_COM = wheel_data['Trial0001'][0][0][5][0][0]['Positions'][0] #(3,t)

segment_data = sci.loadmat(f'./data/Kinematics + EMG/Trial {trial} Segment CoM.mat') #body segments - arm, leg, ect.

# axis = 0 #0 for x, 1, for y, 2 for z
axis_labels = ['horizontal in wheel plane','perpendicular to wheel plane','vertical in wheel plane']
# Printing the keys in our .mat file. 
# print(mat_data.keys())  
keys = list(segment_data.keys())
seg_keys = keys[6:] 



time_passed = len(segment_data[keys[-1]][0][0])
time = range(time_passed)

# Wheel center of mass coordinates
def COM_coordinates(v, rot, com):
    array = []
    rot_mat = np.transpose(rot, axes=(0, 2, 1))

    for t in range(np.shape(v)[0]):
        
        array.append(np.dot(rot_mat[t].T,  v[t]- com[t]))
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


R = scirot.Rotation.from_matrix(rot_mat)
euler = R.as_euler("zxy")
print(np.shape(euler))
euler_cut = euler[start_frame:end_frame,:]

time = np.linspace(start_frame/fps,np.shape(euler_cut)[0]/fps,np.shape(euler_cut)[0] )


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
COM = np.zeros(shape = np.shape(COM_segs[0][start_frame:end_frame]))
for i in range(len(body_segment_masses)):
    COM += body_segment_masses[i]*COM_segs[i][start_frame:end_frame]
COM = COM / 70 #normalize by 70kg


''' Plot waveforms and FFTs '''

fig, ax = plt.subplots(2,4, figsize = (16,6))

for i in range(3):
    # Plot waveforms
    ax[0, i].plot(time, raddeg*euler_cut[:,i])
    centered  =euler_cut[:,i] - np.mean(euler_cut[:,i])
    fft_result = np.fft.fft(centered)  # FFT of the principal component
    N = len(centered)  # Number of samples
    frequencies = np.fft.fftfreq(N, 1 / fps)  # Frequency in Hz (1/seconds)
    magnitude = np.abs(fft_result)
    half_N = N // 2  # Only take the positive frequencies

    max_index= np.argmax(magnitude[:half_N])
    max_freq = frequencies[max_index]
    print(f"Highest Freq: {max_freq}")
    ax[1,i].plot(frequencies[:half_N], magnitude[:half_N] /np.sum(magnitude[:half_N]))  
    ax[1,i].axvline(max_freq, linestyle = '--', label = np.round(max_freq,2))
    ax[0,i].set_title(['Spin (ψ)','Tilt(θ)','Roll(ϕ)'][i])
    ax[1,i].set_xlim(0,4)  
    ax[1,i].legend()
ax[0,-1].plot(time, COM.T[1])
signal = COM.T[1] - np.mean(COM.T[1])
fft_result = np.fft.fft(signal)  # FFT of the signal
frequencies = np.fft.fftfreq(len(signal), 1 / fps)  # Corresponding frequencies

# Compute the magnitude spectrum
magnitude = np.abs(fft_result)
max_index= np.argmax(magnitude[:half_N])
max_freq = frequencies[max_index]
ax[1,-1].plot(frequencies[:fps // 2], magnitude[:fps // 2] / np.sum(magnitude[:fps // 2]))
ax[0,-1].set_title('c(t)')
ax[1,-1].set_xlim(0,4)  
ax[1,-1].axvline(max_freq, linestyle = '--', label = np.round(max_freq,2))

ax[0,-1].set_xlabel("Time (s)")
ax[0,-1].set_ylabel("Amplitude (m)")
ax[0,0].set_ylabel("Angle (degs)")
ax[1,0].set_ylabel("Magnitude")
ax[1,-1].set_xlabel("Frequency (Hz)")
ax[1,-1].legend()

plt.tight_layout()

plt.savefig(f'Trial{trial}_analysis/PCA/motion/rotCOM.png')
plt.clf()

''' Plot FFTs overlayed '''

fig2, ax = plt.subplots(3,1)
for i in range(3):
    signal = COM.T[1] - np.mean(COM.T[1])
    fft_result = np.fft.fft(signal)  # FFT of the signal
    frequencies = np.fft.fftfreq(len(signal), 1 / fps)  # Corresponding frequencies

    # Compute the magnitude spectrum
    magnitude = np.abs(fft_result)
    ax[i].plot(frequencies[:fps // 2], magnitude[:fps // 2] / np.sum(magnitude[:fps // 2]), label = "COM")



    #Rotations
    centered  =euler_cut[:,i] - np.mean(euler_cut[:,i])
    fft_result = np.fft.fft(centered)  # FFT of the principal component
    N = len(centered)  # Number of samples
    frequencies = np.fft.fftfreq(N, 1 / fps)  # Frequency in Hz (1/seconds)
    magnitude = np.abs(fft_result)
    half_N = N // 2  # Only take the positive frequencies

    max_index= np.argmax(magnitude[:half_N])
    max_freq = frequencies[max_index]
    print(f"Highest Freq: {max_freq}")
    ax[i].plot(frequencies[:half_N], magnitude[:half_N] /np.sum(magnitude[:half_N]) )  
    ax[i].set_xlabel("Frequency (Hz)")
    ax[i].set_ylabel("Magnitude")
    ax[i].set_title(['z','x','y'][i])
    ax[i].set_xlim(0,4)

plt.legend()
plt.tight_layout()
plt.savefig(f"Trial{trial}_analysis/PCA/motion/rotCOMfft")
# plt.show()