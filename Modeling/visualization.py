import time 
import itertools 
import numpy as np
import mediapy as media
import matplotlib.pyplot as plt
import mujoco
# import distutils.util
import os
import subprocess
np.set_printoptions(precision=3, suppress=True, linewidth=100)

# SETTING BASIC PROPERTIES
xml_path = "cyr_wheel.xml" # Setting the path for the model
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path) # making sure that it aligns with our directory
xml_path = abspath
simulation_time = 1000

model = mujoco.MjModel.from_xml_path(xml_path) # this makes a model out of the xml such that we can use it. 
data = mujoco.MjData(model)

# Make renderer, render and show the pixels
renderer = mujoco.Renderer(model)
# media.show_image(renderer.render())

mujoco.mj_forward(model, data)
renderer.update_scene(data, "track")

# media.show_image(renderer.render())

def render(xml, duration, framerate, camera):
    model = mujoco.MjModel.from_xml_path(xml) # this makes a model out of the xml such that we can use it. 

    data = mujoco.MjData(model)

    # Make renderer, render and show the pixels
    renderer = mujoco.Renderer(model)
    # media.show_image(renderer.render())

    mujoco.mj_forward(model, data)
    renderer.update_scene(data)

    #media.show_image(renderer.render())

    # Simulate and display video.
    frames = []
    mujoco.mj_resetDataKeyframe(model, data, 0)  # Reset the state to keyframe 0
    while data.time < duration:
        mujoco.mj_step(model, data)
        if len(frames) < data.time * framerate:
            renderer.update_scene(data, camera)
            pixels = renderer.render()
            frames.append(pixels)

    # media.show_video(frames, fps=framerate)
    media.write_video('v.mp4', frames, fps=60, qp=18)

render(xml_path, 7, 20, "track")