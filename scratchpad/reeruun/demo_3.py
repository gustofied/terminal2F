import math

import numpy as np
import rerun as rr


def main():
    rr.init("step3_3d_transforms", spawn=True)

    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("sun", rr.Points3D([0.0, 0.0, 0.0], radii=1.0, colors=[255, 200, 10]))
    rr.log("sun/planet", rr.Points3D([0.0, 0.0, 0.0], radii=0.4, colors=[40, 80, 200]))
    rr.log(
        "sun/planet/moon",
        rr.Points3D([0.0, 0.0, 0.0], radii=0.15, colors=[180, 180, 180]),
    )
    rr.log(
        "sun/planet/moon2",
        rr.Points3D([0.0, 0.0, 0.0], radii=0.12, colors=[150, 120, 100]),
    )

    # log the circle

    input("Press Enter to continue...")

    d_planet = 6.0
    d_moon = 1.75
    d_moon2_x = 2.5
    d_moon2_y = 1.5
    angles = np.arange(0.0, 1.01, 0.01) * np.pi * 2
    circle = np.array(
        [np.sin(angles), np.cos(angles), angles * 0.0], dtype=np.float32
    ).transpose()

    # Create oval path for second moon
    oval = np.array(
        [np.sin(angles) * d_moon2_x, np.cos(angles) * d_moon2_y, angles * 0.0],
        dtype=np.float32,
    ).transpose()
    rr.log(
        "sun/planet_path", rr.LineStrips3D(circle * d_planet, colors=[100, 100, 255])
    )

    rr.log(
        "sun/planet/moon_path", rr.LineStrips3D(circle * d_moon, colors=[200, 200, 200])
    )

    rr.log("sun/planet/moon2_path", rr.LineStrips3D(oval, colors=[150, 120, 100]))
    # log the circle
    input("Press Enter to continue...")
    total_frames = 6 * 360

    for frame in range(total_frames):
        time = frame / 120.0

        r_planet = time * 2.0
        r_moon = time * 5.0
        r_moon2 = time * 3.5

        planet_x = math.sin(r_planet) * d_planet
        planet_y = math.cos(r_planet) * d_planet
        planet_z = 0.0

        rr.log(
            "sun/planet",
            rr.Transform3D(
                translation=[planet_x, planet_y, planet_z],
                rotation=rr.RotationAxisAngle(axis=(1, 0, 0), degrees=20),
            ),
        )

        moon_x = math.cos(r_moon) * d_moon
        moon_y = math.sin(r_moon) * d_moon
        moon_z = 0.0

        rr.log(
            "sun/planet/moon",
            rr.Transform3D(
                translation=[moon_x, moon_y, moon_z],
                relation=rr.TransformRelation.ChildFromParent,
            ),
        )

        moon2_x = math.sin(r_moon2) * d_moon2_x
        moon2_y = math.cos(r_moon2) * d_moon2_y
        moon2_z = 0.0

        rr.log(
            "sun/planet/moon2",
            rr.Transform3D(
                translation=[moon2_x, moon2_y, moon2_z],
                relation=rr.TransformRelation.ChildFromParent,
            ),
        )

        if frame == 0:
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()