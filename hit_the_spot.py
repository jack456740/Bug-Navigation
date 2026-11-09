
import time
import numpy as np
from mbot_bridge.api import MBot


def normalize_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def drive_to_pose(goal_x, goal_y, goal_theta, robot):
    """Drive the robot to [goal_x, goal_y, goal_theta]."""

    position_tolerance = 0.05
    angle_tolerance = 0.05

    linear_speed = 0.15
    angular_speed = 0.5

    # Timeout for reaching the goal position
    start_time = time.monotonic()

    while True:
        if time.monotonic() - start_time > 60:
            robot.stop()
            raise TimeoutError("Failed to reach goal position")

        # Read current position
        x, y, theta = robot.read_odometry()

        # Calculate distance to goal
        dx = goal_x - x
        dy = goal_y - y
        distance = np.sqrt(dx**2 + dy**2)

        # Stop moving forward when close enough
        if distance < position_tolerance:
            break

        # Calculate angle toward goal
        target_angle = np.arctan2(dy, dx)
        angle_error = normalize_angle(target_angle - theta)

        # Turn toward goal first
        if abs(angle_error) > angle_tolerance:
            robot.drive(
                0.0,
                0.0,
                angular_speed * np.sign(angle_error)
            )
        else:
            # Move forward
            robot.drive(linear_speed, 0.0, 0.0)

        time.sleep(0.1)

    # Stop before final rotation
    robot.stop()

    # Rotate to goal orientation
    rotation_start_time = time.monotonic()

    while True:
        if time.monotonic() - rotation_start_time > 30:
            robot.stop()
            raise TimeoutError("Failed to reach goal orientation")

        x, y, theta = robot.read_odometry()
        angle_error = normalize_angle(goal_theta - theta)

        if abs(angle_error) < angle_tolerance:
            break

        robot.drive(
            0.0,
            0.0,
            angular_speed * np.sign(angle_error)
        )

        time.sleep(0.1)

    robot.stop()

    # Report final position and orientation errors
    x, y, theta = robot.read_odometry()

    position_error = np.hypot(goal_x - x, goal_y - y)
    orientation_error = abs(normalize_angle(goal_theta - theta))

    print("Position error:", position_error, "meters")
    print("Orientation error:", orientation_error, "radians")


robot = MBot()

try:
    # Reset odometry before beginning
    robot.reset_odometry()

    # P2.1: Drive to a sequence of goal poses
    drive_to_pose(1, 0, 0, robot)
    time.sleep(1)

    drive_to_pose(1, 1, np.pi / 2, robot)
    time.sleep(1)

    drive_to_pose(0, 1, np.pi, robot)
    time.sleep(1)

    drive_to_pose(0, 0, 0, robot)

finally:
    # Stop robot before exiting
    robot.stop()

    # Print final odometry
    print("Final odometry:", robot.read_odometry())


