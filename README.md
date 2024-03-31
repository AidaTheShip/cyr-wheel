# cyr-wheel

# TO-DO 
- Figure out how units work in Mujoco.
- Reinforcement Learning Algorithm in Mujoco. 
- Scale/Rotate/Translate objects in Mujoco. (Intial conditions)

# Control Algorithm 
Add more information here...

# Reinforcement Learning 
We will use Mujuco to model the dynamics of the wheel in python. Then we use stable baselines to provide and set up a stable reinforcement learning algorithm. 

## Mujoco
Make sure that you install Mujuco: pip install mujoco 
Note that everything in Mujuco is in cartesian coordinates. 

Mujuco uses XML files. What are XML files? 
XML files lets you define and store data in a shareable manner. It provides rules to define any data. It cannot perform its own computing operations but it can be implemented for structured data managemnet. 

### Modeling
[https://www.youtube.com/playlist?list=PLc7bpbeTIk75dgBVd07z6_uKN1KQkwFRK]

### Installing ffmpeg for video rendering
Follow this guide: https://phoenixnap.com/kb/ffmpeg-mac (involves installing Homebrew)

### d.o.f
qpos uses quaternions for position (see https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation for math). The first three d.o.f are x,y,z, and the four last ones are the four quaternion d.o.f associated with rotations.
qvel = [x y z ?psi? theta phi] (see paper in dropbox for drawing showing these angles). psi does not have an affect at small vel, leads to turbulence at high vel

