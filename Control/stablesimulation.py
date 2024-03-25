# importing all the necessary modules
import numpy as np
import sympy
import casadi 
import modules as M
import os
import matplotlib.pyplot as plt

colorsBlue = ['#b3e5fc', '#0091ea']
colorsRed = ['#f18e86', '#e83b47']
colors = [colorsBlue, colorsRed]

m = 5
casadi_EOM, x, y = M.get_casadi_EOM(m)

energy_EQN = M.energy(m)

ode = {}
ode['x'] = x
ode['p'] = u
ode['ode'] = casadi.vertcat(*casadi_EOM)

dt = 1e-2
t = np.arange(0,dt*2000, dt)

F = casadi.integrator('F', 'cvodes', ode, 0, t)

# Stable general mode. inputs are: theta0, psi1 (cannot be 0!), m, and c0

x0 = M.get_stable_mode(np.pi/2 - 0.5, 0.5, m, 0.4)
print(x0)

# Stable travelling mode

x0 = [0, np.pi/2 - 0.6, 0,
      0,0,3,
      0,0,
      0.820964,0]

print(x0)

# Stable spinning mode

x0 = [0, np.pi/2 - 0.3, 0,
      0.815932,0,0,
      0,0,
      0.38,0]

print(x0)

res = F(x0 = x0, p = 0)
# print(res["xf"])
traj_ss = np.array(res["xf"]).T # ss = stable state
# print(traj_ss)

traj_CT = []
traj_GC = []
traj_EE = []
for i in range(len(traj_ss)):
    traj_CT.append(M.CT(traj_ss[i]))
    traj_GC.append(M.GC(traj_ss[i]))
    traj_EE.append(energy_EQN(*traj_ss[i])) # EE might be the energy expressed in terms of the trajectory in the lab frame
traj_CT = np.array(traj_CT)
traj_GC = np.array(traj_GC)
traj_EE = np.array(traj_EE)
    
    
# plot the ConTact point trajectory and the Geometric Centre trajectory
plt.figure()
plt.axes().set_aspect(1)
plt.plot(traj_CT.T[0], traj_CT.T[1], color = colors[0][1], label = 'contact point')
plt.plot(traj_GC.T[0], traj_GC.T[1], color = colors[1][1], linestyle = 'dashed', label = 'geometric centre')

plt.legend()

# plot the energy over time
plt.figure()
plt.plot(np.arange(0,dt*(len(traj_ss) - 0.5),dt), traj_EE)
plt.savefig("")


# plots of Euler angles and angular velocities

fig, axes = plt.subplots(2, 3, figsize = (9, 6), sharex = True)
axes = np.reshape(axes, -1)

qty = ['psi0', 'theta0', 'phi0', 'psi1', 'theta1', 'phi1']

for i in range(6):
    axes[i].plot(np.arange(0,dt*(len(traj_ss) - 0.5),dt), traj_ss.T[i], color = colors[0][1])
    axes[i].set_title(qty[i])
    
ax = plt.figure().add_subplot(projection='3d')

os.makedirs('frames', exist_ok = True)
skip_N = 20

for i in range(len(traj_ss)):

    if i % skip_N == 0:
        print("Progress: {0:.1f}%".format(100 * i/len(traj_ss)), end = '\r')

        M.DrawCircle(ax, traj_ss[i])

        ax.plot(np.array(traj_GC[:i,0]),
                np.array(traj_GC[:i,1]),
                np.array(traj_GC[:i,2]),
                color = 'red', zorder = 1)

        ax.plot(np.array(traj_CT[:i,0]),
                np.array(traj_CT[:i,1]),
                np.array(traj_CT[:i,2]),
                color = 'black', zorder = 1)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        ax.set_xlim(traj_GC[i,0] - 3, traj_GC[i,0] + 3)
        ax.set_ylim(traj_GC[i,1] - 3, traj_GC[i,1] + 3)
        #ax.set_xlim(- 4, + 4)
        #ax.set_ylim(- 4, + 4)
        ax.set_zlim(0, 6)

        plt.savefig('frames/stable/{0:05d}.png'.format(i//skip_N))

        ax.clear()