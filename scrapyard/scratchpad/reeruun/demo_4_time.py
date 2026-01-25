import math
import time

import rerun as rr


def main():
    rr.init("step4_multiple_timelines", recording_id="example_timeline")
    # expects to be run from the gentle-intro directory
    rr.save("data/example_timeline/step4.rrd")

    start_time = time.time()

    for frame in range(60):
        current_time = start_time + frame * 0.1

        rr.set_time("imu_time", sequence=frame)
        rr.set_time("simulation_time", timestamp=current_time)

        imu_angle_x = frame * 0.08
        imu_angle_y = frame * 0.05
        imu_angle_z = frame * 0.12

        orientation_points = []
        for i in range(3):
            x = i * 0.5 - 0.5
            y = math.sin(imu_angle_x) * 0.3
            z = math.cos(imu_angle_y) * 0.3

            new_x = x * math.cos(imu_angle_z) - y * math.sin(imu_angle_z)
            new_y = x * math.sin(imu_angle_z) + y * math.cos(imu_angle_z)

            orientation_points.append([new_x, new_y, z])

        rr.log(
            "drone/imu_orientation",
            rr.Points3D(orientation_points, colors=[[255, 100, 100]], radii=0.08),
        )

        if frame % 5 == 0:
            gps_frame = frame // 5
            rr.set_time("gps_time", sequence=gps_frame)
            rr.set_time("simulation_time", timestamp=current_time)

            gps_angle = gps_frame * 0.3
            gps_x = math.cos(gps_angle) * 2
            gps_y = math.sin(gps_angle) * 2
            gps_z = 0.5 + math.sin(gps_frame * 0.2) * 0.3

            rr.log(
                "drone/gps_position",
                rr.Points3D(
                    [[gps_x, gps_y, gps_z]], colors=[[100, 255, 100]], radii=0.12
                ),
            )

        if frame % 10 == 0:
            obstacle_frame = frame // 10
            rr.set_time("obstacle_time", sequence=obstacle_frame)
            rr.set_time("simulation_time", timestamp=current_time)

            obstacle_points = []
            obstacle_colors = []

            for i in range(3):
                obs_angle = obstacle_frame * 0.4 + i * 2.1
                obs_distance = 3 + i * 0.5

                obs_x = math.cos(obs_angle) * obs_distance
                obs_y = math.sin(obs_angle) * obs_distance
                obs_z = 0.2 + i * 0.3

                obstacle_points.append([obs_x, obs_y, obs_z])
                obstacle_colors.append([255, 150, 50])

            rr.log(
                "environment/detected_obstacles",
                rr.Points3D(obstacle_points, colors=obstacle_colors, radii=0.15),
            )

        rr.set_time("frame", sequence=frame)


if __name__ == "__main__":
    main()