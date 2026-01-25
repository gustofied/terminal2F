from rerun.components import TextLogLevel
from rerun_bindings import spawn
import numpy as np
import rerun as rr
import matplotlib.pyplot as plt
import rerun.blueprint as rrb
plt.style.use(['dark_background'])

# x = np.arange(0, 10 ,1)
# print(x)

# y = np.linspace(0, 1, 20)
# print(y)

# f = x**2 * np.sin(x*10)

# plt.figure(figsize=(4,2))
# plt.plot(x, f)
# plt.show()



# b = np.linspace(0, 1, 11)
# c = np.linspace(0, 1, 20)

# print("..")

# bc = np.meshgrid(b,c)
# print(bc)

# plt.figure(figsize=(4,2))
# x, y = bc
# plt.plot(x, y)
# plt.show()

print("- - - - - - - - - - - - ")


# h= np.linspace(0, 7, 8)
# j = np.linspace(0, 3, 4)

# for xpoint in h:
#     for ypoint in j:
#         plt.plot(xpoint, ypoint, "wh")

# plt.show()


# print("- - - - - - - - - - - - ")


# x2d, y2d = np.meshgrid(h, j)

# plt.plot(x2d, y2d, "yh")
# plt.show()

# rr.init("hey_im_learning", spawn=True)
# rr.log("dd,",rr.Mesh3D(
#     f,

# ), )



rr.init("Learning_1", spawn=True)

rr.set_time("time", sequence=0)
rr.log("log/status", rr.TextLog("Application started ", level=rr.TextLogLevel.INFO))
rr.set_time("time", sequence=5)
rr.log("log/status", rr.TextLog("A warning", level=rr.TextLogLevel.WARN))
for i in range(10):
    rr.set_time("time", sequence=i)
    rr.log(
        "log/status", rr.TextLog(f"Proccesed item {i}", level= rr.TextLogLevel.INFO)
    )


# build a plueprint then send it??

blueprint = rrb.Blueprint(
    rrb.TextLogView(origin="/log", name="Text Log"),
    rrb.TimePanel(state="expanded"),
    collapse_panels=True
)
rr.send_blueprint(blueprint)


