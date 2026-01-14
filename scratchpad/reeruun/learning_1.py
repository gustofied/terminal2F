import numpy as np
import rerun as rr

rr.init("Hey", spawn=True)

rr.log(
    "my/points",
    rr.Points3D([[0.2, 0.5, 0.3], [0.9, 1.2, 0.1], [1.0, 4.2, 0.3]], radii=[0.1, 0.2, 0.3]),
    rr.Arrows3D(vectors=[[0.3, 2.1, 0.2], [0.9, -1.1, 2.3], [-0.4, 0.5, 2.9]]),
    rr.AnyValues(confidence=[0.3, 0.4, 0.9]),
)