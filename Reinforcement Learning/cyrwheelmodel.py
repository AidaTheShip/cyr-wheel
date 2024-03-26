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
        <mesh name="torus_mesh" file="torus.obj"/>
    </asset>
    <worldbody>
        <body name="torus_body" pos="0 0 0">
            <geom type="mesh" mesh="torus_mesh" size="1 1 1"/>
        </body>
    </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml) # this makes a model out of the xml such that we can use it. 