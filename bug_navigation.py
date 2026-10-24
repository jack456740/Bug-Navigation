
import time
import math
import numpy as np
from mbot_bridge.api import MBot

robot = MBot()

# Navigation settings
FORWARD_SPEED = 0.15
TURN_SPEED = 0.6
SAFE_DISTANCE = 0.35
POSITION_TOLERANCE = 0.10
ANGLE_TOLERANCE = 0.10

# Reset odometry
robot.reset_odometry()

# P2.4: Ask user for goal pose
goal_x = float(input("Enter goal x: "))
goal_y = float(input("Enter goal y: "))
goal_theta = float(input("Enter goal angle (radians): "))


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def find_min_dist(ranges, thetas):
    min_dist = None
    min_angle = None

    for dist, angle in zip(ranges, thetas):
        if dist <= 0:
            continue

        if min_dist is None or dist < min_dist:
            min_dist = dist
            min_angle = angle

    return min_dist, min_angle


try:
    while True:
        # Replace these calls with the methods provided
        # by your MBot bridge API.
        odom = robot.read_odometry()
        scan = robot.read_lidar()

        x = odom.x
        y = odom.y
        theta = odom.theta

        dx = goal_x - x
        dy = goal_y - y

        distance = math.hypot(dx, dy)

        # Stop translating when close to the goal
        if distance < POSITION_TOLERANCE:
            angle_error = normalize_angle(
                goal_theta - theta
            )

            if abs(angle_error) < ANGLE_TOLERANCE:
                break

            robot.drive(0.0, 0.0,
                        TURN_SPEED * np.sign(angle_error))
            time.sleep(0.1)
            continue

        # Check for nearby obstacles
        min_dist, min_angle = find_min_dist(
            scan.ranges, scan.thetas
        )

        if min_dist is not None and min_dist < SAFE_DISTANCE:
            # Turn away from obstacle
            turn = -TURN_SPEED if min_angle > 0 else TURN_SPEED
            robot.drive(0.0, 0.0, turn)

        else:
            # Turn toward goal
            target_angle = math.atan2(dy, dx)
            angle_error = normalize_angle(
                target_angle - theta
            )

            if abs(angle_error) > 0.15:
                robot.drive(
                    0.0, 0.0,
                    TURN_SPEED * np.sign(angle_error)
                )
            else:
                robot.drive(
                    FORWARD_SPEED, 0.0, 0.0
                )

        time.sleep(0.1)

finally:
    robot.stop()

    # Print final odometry
    final_odom = robot.read_odometry()
    print("Final odometry:", final_odom)

