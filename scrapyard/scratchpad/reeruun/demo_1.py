import numpy as np
import rerun as rr

def main():

    rr.init("demo_1_hello_rerun", spawn=True)
    instructions = """
    Demo 1; Hello Rerun
    What's happening:
    1. We init rerun
    2. Generate some points
    3. We log these to the viewer
    4. when running it auto open since we got spawn = True
    """

    rr.log(
        "instructions",
        rr.TextDocument(instructions, media_type=rr.MediaType.MARKDOWN),
        static=True,
    )

    rng = np.random.default_rng(6)
    a_random_number = rng.integers(10)

    n_bubbles = 10
    positions_d3 = rng.uniform(-3, 3, (n_bubbles, 3))
    print(positions_d3)
    positions_d2 = positions_d3[:, :2]
    print(positions_d2)

    colors = rng.uniform(0, 255, size=[n_bubbles, 4])
    radii = rng.uniform(0, 1, size=[n_bubbles])

    rr.log("points/2d",rr.Points2D(positions_d2, colors=colors, radii=radii))
    rr.log("points/3d",rr.Points3D(positions_d3, colors=colors, radii=radii))

    rr.log(
        "instructions/number_example",
        rr.TextDocument(str(a_random_number), media_type=rr.MediaType.MARKDOWN),
        static=True,
    )

if __name__ == "__main__":
    main()



