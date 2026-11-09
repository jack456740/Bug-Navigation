import time
import numpy as np
from mbot_bridge.api import MBot

def obstacle_ahead(ranges, thetas, threshold):
    for distance, angle in zip(ranges, thetas):
        if not np.isfinite(distance) or distance <= 0 or not np.isfinite(angle):
            continue

        if abs(angle) < np.pi / 6 and distance < threshold:
            return True

    return False


def find_wall_distance(ranges, thetas):
    wall_distances = []

    for distance, angle in zip(ranges, thetas):
        if not np.isfinite(distance) or distance <= 0 or not np.isfinite(angle):
            continue

        if -np.pi / 2 - 0.2 <= angle <= -np.pi / 2 + 0.2:
            wall_distances.append(distance)

    if wall_distances:
        return min(wall_distances)

    return None


robot = MBot()

# Navigation states
GO_TO_GOAL = 0
FOLLOW_WALL = 1
GOAL_REACHED = 2

# Initial navigation state
state = GO_TO_GOAL
previous_state = None

state_names = {
    GO_TO_GOAL: "GO_TO_GOAL",
    FOLLOW_WALL: "FOLLOW_WALL",
    GOAL_REACHED: "GOAL_REACHED"
}
# Distance tolerance in meters
position_tolerance = 0.05
linear_speed = 0.15
angular_speed = 0.5
angle_tolerance = 0.05
obstacle_distance = 0.40
desired_wall_distance = 0.3048
wall_kp = 1.0
wall_distance = None
hit_x = None
hit_y = None
hit_distance = None
previous_path_side = None


# Reset the robot odometry to zero at the beginning of the run.
robot.reset_odometry()
start_x, start_y, start_theta = robot.read_odometry()

if not all(np.isfinite(value) for value in (start_x, start_y, start_theta)):
    robot.stop()
    raise RuntimeError("Invalid starting odometry readings")


goal_x = float(input("Enter goal X (meters): "))
goal_y = float(input("Enter goal Y (meters): "))
goal_theta = float(input("Enter goal angle (radians): "))
if not all(np.isfinite(value) for value in (goal_x, goal_y, goal_theta)):
    raise ValueError("Goal coordinates and angle must be finite numbers")

print("Goal pose:", goal_x, goal_y, goal_theta)
start_time = time.monotonic()
try:
    while state != GOAL_REACHED:
        if state != previous_state:
            print("Current state:", state_names[state])
            previous_state = state
        if time.monotonic() - start_time > 60:
            robot.stop()
            raise TimeoutError("Navigation timed out")
        x, y, theta = robot.read_odometry()
        if not all(np.isfinite(value) for value in (x, y, theta)):
            robot.stop()
            raise RuntimeError("Invalid robot odometry readings")
        ranges, thetas = robot.read_lidar()
        if len(ranges) != len(thetas):
            robot.stop()
            raise RuntimeError("LiDAR ranges and angles have different lengths")
        valid_front_readings = any(
            np.isfinite(distance)
            and distance > 0
            and np.isfinite(angle)
            and abs(angle) < np.pi / 6
            for distance, angle in zip(ranges, thetas)
        )

        if not valid_front_readings:
            robot.stop()
            raise RuntimeError("No valid forward LiDAR readings")
        wall_distance = find_wall_distance(ranges, thetas)
        dx = goal_x - x
        dy = goal_y - y
        distance = np.sqrt(dx**2 + dy**2)
        target_angle = np.arctan2(dy, dx)
        angle_error = target_angle - theta
        angle_error = np.arctan2(
            np.sin(angle_error),
            np.cos(angle_error)
        )
        final_angle_error = goal_theta - theta
        final_angle_error = np.arctan2(
            np.sin(final_angle_error),
            np.cos(final_angle_error)
        )
        print("Distance to goal:", distance)
        print("Angle error:", angle_error)
        print("Final angle error:", final_angle_error)
        if (distance < position_tolerance and
            abs(final_angle_error) < angle_tolerance):
            state = GOAL_REACHED
            robot.stop()
            print("Current state:", state_names[state])
            print("Goal pose reached!")
        elif state == GO_TO_GOAL:
            if distance < position_tolerance:
                turn_speed = np.clip(
                    1.5 * final_angle_error,
                    -angular_speed,
                    angular_speed
                )

                print("Rotate to final angle:", turn_speed)
                robot.drive(0.0, 0.0, float(turn_speed))

            elif obstacle_ahead(ranges, thetas, obstacle_distance):
                hit_x = x
                hit_y = y
                hit_distance = distance
                previous_path_side = None

                state = FOLLOW_WALL
                robot.stop()
                print("Obstacle detected! Switching to FOLLOW_WALL")
                print("Hit position:", hit_x, hit_y)

            elif abs(angle_error) > angle_tolerance:
                turn_speed = np.clip(
                    1.5 * angle_error,
                    -angular_speed,
                    angular_speed
                )

                print("Turn speed:", turn_speed)
                robot.drive(0.0, 0.0, float(turn_speed))
            else:
                print("Move forward at speed:", linear_speed)
                robot.drive(linear_speed, 0.0, 0.0)
                
        elif state == FOLLOW_WALL:
            print("Following obstacle wall")
            closer_to_goal = (
                hit_distance is not None
                and distance < hit_distance - 0.05
            )

            print("Closer to goal than hit point:", closer_to_goal)
            path_dx = goal_x - start_x
            path_dy = goal_y - start_y

            path_length = np.hypot(path_dx, path_dy)
            path_side = (
                path_dx * (y - start_y)
                  - path_dy * (x - start_x)
                  )
            if previous_path_side is None:
                previous_path_side = path_side

            if path_length > 0:
                path_error = abs(
                    path_dy * (x - start_x)
                    - path_dx * (y - start_y)
                ) / path_length
            else:
                path_error = float("inf")

            print("Distance from original path:", path_error)
            crossed_path = (
                previous_path_side * path_side <= 0
                and previous_path_side != path_side
            )

            previous_path_side = path_side
            if (
                closer_to_goal
                and crossed_path
                and np.hypot(x - hit_x, y - hit_y) > 0.15
                and not obstacle_ahead(
                    ranges,
                    [
                        np.arctan2(
                            np.sin(angle + theta - target_angle),
                            np.cos(angle + theta - target_angle)
                            )
                            for angle in thetas
                            ],
                            obstacle_distance
                            )
                        ):
                state = GO_TO_GOAL
                robot.stop()
                print("Back on original path. Returning to GO_TO_GOAL.")

            elif wall_distance is not None:
                wall_error = wall_distance - desired_wall_distance
                print("Wall distance error:", wall_error)
                wall_turn_speed = np.clip(
                    -wall_kp * wall_error,
                    -angular_speed,
                    angular_speed
                )

                print("Wall turn speed:", wall_turn_speed)
                if obstacle_ahead(ranges, thetas, obstacle_distance):
                    robot.drive(
                        0.0,
                        0.0,
                        angular_speed
                    )
                    print("Obstacle ahead. Turning left.")
                else:
                    robot.drive(
                        linear_speed,
                        0.0,
                        float(wall_turn_speed)
                    )
            else:
                robot.stop()
                raise RuntimeError("Lost right-side wall during navigation")

        time.sleep(0.1)
finally:
    robot.stop()


final_x, final_y, final_theta = robot.read_odometry()

print("Final X:", final_x)
print("Final Y:", final_y)
print("Final Theta:", final_theta)