import time 
import itertools 
import numpy as np
import mediapy as media
import matplotlib.pyplot as plt
import mujoco

np.set_printoptions(precision=3, suppress=True, linewidth=100)

# Defining the wheel here...
xml = """
<mujoco>
    <asset>
        <mesh name="cyrwheel_mesh" file="cyrwheel.obj"/>
    </asset>
    <worldbody>
        <body name="cyrwheel" pos="0 0 0">
            <geom type="mesh" mesh="cyrwheel_mesh" size="1 1 1"/>
        </body>
    </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml) # this makes a model out of the xml such that we can use it. 

data = mujoco.MjData(model)

# Make renderer, render and show the pixels
renderer = mujoco.Renderer(model)
media.show_image(renderer.render())

mujoco.mj_forward(model, data)
renderer.update_scene(data)

media.show_image(renderer.render())