
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

    while True:
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
    while True:
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


robot = MBot()

# Reset odometry
robot.reset_odometry()

try:
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
