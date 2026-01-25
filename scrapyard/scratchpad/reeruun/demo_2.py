from ctypes import pointer
import math

import rerun as rr

# rotation matixy

def new_location(center_x, center_y, x, y, angle):
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    # Translate to origin
    translated_x = x - center_x
    translated_y = y - center_y

    # Rotate
    new_x = translated_x * cos_a - translated_y * sin_a + center_x
    new_y = translated_x * sin_a + translated_y * cos_a + center_y

    return new_x, new_y

def main():
    rr.init("step2_points_animation_colors", spawn=True)

    square_points = [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
    ]
    center = [0.5, 0.5]
    point_names = [f"Point_{x}" for x in range(4)]
    rr.log(
        "animated_square/origin",
        rr.Points2D(
            [center],
            colors=[[0.4, 0.2, 0.2]],
            radii=[0.5],
            show_labels=True,
            labels=["Origin"],
        ),
        static=True,
    )
    for frame in range(360):
        rr.set_time("frame", sequence=frame)

        angle = frame * 0.05

        rotated_points = []
        for point in square_points:
            new_x, new_y = new_location(center[0], center[1], point[0], point[1], angle)
            rotated_points.append([new_x, new_y])

        dynamic_colors = []
        for i in range(len(square_points)):
            hue = (frame * 2 + i * 90) % 360

            c = 1.0
            x_color = c * (1 - abs((hue / 60) % 2 - 1))

            if 0 <= hue < 60:
                r, g, b = c, x_color, 0
            elif 60 <= hue < 120:
                r, g, b = x_color, c, 0
            elif 120 <= hue < 180:
                r, g, b = 0, c, x_color
            elif 180 <= hue < 240:
                r, g, b = 0, x_color, c
            elif 240 <= hue < 300:
                r, g, b = x_color, 0, c
            else:
                r, g, b = c, 0, x_color

            dynamic_colors.append([int(r * 255), int(g * 255), int(b * 255)])

        dynamic_radii = []
        for i in range(len(square_points)):
            base_size = 0.04
            pulse_amplitude = 0.025
            pulse_rate = 0.15 + i * 0.03

            pulse_factor = math.sin(frame * pulse_rate)
            radius = base_size + pulse_amplitude * pulse_factor
            dynamic_radii.append(max(radius, 0.01))

        rr.log(
            "animated_square/points",
            rr.Points2D(
                rotated_points,
                colors=dynamic_colors,
                radii=dynamic_radii,
                labels=point_names,
                show_labels=True,
            ),
        )

        for i, (point, name) in enumerate(zip(rotated_points, point_names)):
            x, y = point

            rr.log(f"position/{name}/x", rr.Scalars(x))
            rr.log(f"position/{name}/y", rr.Scalars(y))

        rr.log("overall/position/rotation_angle_radians", rr.Scalars(angle))


if __name__ == "__main__":
    main()


