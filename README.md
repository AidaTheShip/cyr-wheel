# cyr-wheel

# TO-DO 
- Figure out how units work in Mujoco.
- Reinforcement Learning Algorithm in Mujoco. 
- Scale/Rotate/Translate objects in Mujoco. (Intial conditions)

# Control Algorithm 
Add more information here...

# Kinematics & Dynamics
**Forward kinematics** is a concept used in robotics and computer animation to calculate the position and orientation of an object's parts based on its joint parameters without considering the forces that cause motion. Essentially, it deals with determining the end position (like the tip of a robotic arm or the hand of an animated character) given the angles and displacements of the joints leading up to it.

**Forward Dynamics** refers particularly to the process of calculating the accelerations of parts of a system given the forces and torques acting on them, and then using this information to predict future states of the system. In comparison to forward kinematics, forward dynamics focuses on determining the position and orientation of parts based on joint parameters without regard to forces. Forward dynamics, on the other hand, is about understanding the physical properties of the system (mass, inertia, etc.), and the external forces applied to it, such as gravity or contact forces. 
In essence, forward dynamics answers the question: "Given the current state of a system (positions, velocities) and the forces applied to it, what will be its state at a future time?" This involves solving the equations of motion for the system, which are typically second-order differential equations derived from Newton's second law of motion (F = ma) or, in the case of rotational motion, from Euler's rotation equations.


# Mujoco Modeling
Make sure that you install Mujuco: pip install mujoco 
Note that everything in Mujuco is in cartesian coordinates. 

Mujuco uses XML files. What are XML files? 
XML files lets you define and store data in a shareable manner. It provides rules to define any data. It cannot perform its own computing operations but it can be implemented for structured data managemnet. 
Let's walk through how we are implementing the model step by step. 
We first define our xml files, each of which have containers that defines classes, e.g. <mesh> through which you can define a custom mesh. 
We define a couple of custom things, one of them being the cyr wheel itself, which we are modeling after a torus. 
The grid of the environment, where most of the simulation is going to run on has a checker structure, like many Mujoco simulations do. 
Within the xml description of our environment, we are also setting the worldbody, including instantiating the actual geometries, light, and camera positions. 

We create our **model** through the xml string: 'model = mujoco.MjModel.from_xml_string(xml)'.

## Simulation 
- 'mujoco.mj_step': Advance simulation, use control callback to obtain external force and control.
- 'mujoco.mj_forward': Forward dynamics: same as mj_step but do not integrate in time. We are using mj_forward in the notebook to quickly simulate the Cyr Wheel. Generally speaking, the forward function calculates the positions and orientations of all bodies based on the joint parameters (angles, distances, etc.). It first determines where everything is in space in an initial "kinematic pass" to then using the "dynamic pass" of calculating velocities and accelerations. This involves considering the masses, inertias, joint forces, actuator forces, and external forces acting on the system. The result is the net force and torque on each body. The function then resolves any constraints present in the system, such as contact forces between bodies, joint limits, or friction. Finally, the forward function integrates the calculated accelerations over time to update the positions (this pushes the simulation forward in time) and the output state gives us the new state of the simulation including updated positions, velocities, and any other relevant state information after the simulation step. 

- 'MjData': data structure holding the simulation state. Find it [here][https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjdata]. 

[https://www.youtube.com/playlist?list=PLc7bpbeTIk75dgBVd07z6_uKN1KQkwFRK]

### Installing ffmpeg for video rendering
Follow this guide: https://phoenixnap.com/kb/ffmpeg-mac (involves installing Homebrew)

### d.o.f
qpos uses quaternions for position (see https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation for math). The first three d.o.f are x,y,z, and the four last ones are the four quaternion d.o.f associated with rotations.
qvel = [x y z ?psi? theta phi] (see paper in dropbox for drawing showing these angles). psi does not have an affect at small vel, leads to turbulence at high vel

# Reinforcement Learning 
We will use Mujuco to model the dynamics of the wheel in python. Then we use stable baselines to provide and set up a stable reinforcement learning algorithm. 
## OpenAI
In order for us to train an RL algorithm on the Mujoco environment we have created, we will use (former?) OpenAI's gym pipeline. 
Many pre-set Mujoco environments are set in the Gym's examples and there is an option of [creating your own custom environments][https://www.gymlibrary.dev/content/environment_creation/].

The documentation regarding using Mujoco environments with the Gym says: 
> There are ten Mujoco environments: Ant, HalfCheetah, Hopper, Humanoid, HumanoidStandup, IvertedDoublePendulum, InvertedPendulum, Reacher, Swimmer, and Walker. All of these environments are stochastic in terms of their initial state, with a Gaussian noise added to a fixed initial state in order to add stochasticity. The state spaces for MuJoCo environments in Gym consist of two parts that are flattened and concatented together: a position of a body part (’mujoco-py.mjsim.qpos’) or joint and its corresponding velocity (’mujoco-py.mjsim.qvel’). Often, some of the first positional elements are omitted from the state space since the reward is calculated based on their values, leaviing it up to the algorithm to infer those hidden values indirectly.

### Creating custom environment in mujoco
You have to follow a certain folder structure, that goes something like this: 
""".txt
Folder A/
    setup.py
    FolderB/
        __init__.py
        envs/
            __init__.py
            NewEnv.py
"""

### Gym.Spaces
This superclass is used to define observation and action spaces. Spaces are crucially used in the gym to define the format of valid **action** and **observations**. In this, they clearly define how an agent can interact with environments, allowign us to work with highly structured data, provide a method to sample random elements. 

An example that is used particularly often is the class 'gym.spaces.Box'. Specifically, a box represents a cartesian product of n closed intervals, each with the form or one of [a, b], (-∞, b], [a, ∞], (-∞, ∞).
You can modify the dimensions by doing the following,e.g.: 
"""python

Box(low=_, high=_, shape=(_, _), dtype=np.float32)

"""