"""
Biwheel (differential-drive) robot pure pursuit with PD steering control (v, w) and ROS2 TwistMsg integration.
Adapted to publish linear and angular velocity commands directly, without separate wheel velocities.
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import numpy as np
import math
import matplotlib.pyplot as plt

show_animation = True

class DiffState:
    """
    Differential-drive robot state using linear (v) and angular (w) velocity.
    """
    def __init__(self, x=0.0, y=0.0, yaw=0.0,
                 v=0.0, w=0.0, dt=0.1):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v        # linear velocity [m/s]
        self.w = w        # angular velocity [rad/s]
        self.pp_dt = dt

    def update(self, v_cmd, w_cmd):
        """
        Update pose given commanded linear and angular velocities.
        """
        self.v = v_cmd
        self.w = w_cmd
        # integrate motion
        self.x += self.v * math.cos(self.yaw) * self.pp_dt
        self.y += self.v * math.sin(self.yaw) * self.pp_dt
        self.yaw += self.w * self.pp_dt

class States:
    """History of states for plotting"""
    def __init__(self):
        self.x, self.y, self.yaw = [], [], []
        self.v, self.w, self.t = [], [], []

    def append(self, t, state: DiffState):
        self.x.append(state.x)
        self.y.append(state.y)
        self.yaw.append(state.yaw)
        self.v.append(state.v)
        self.w.append(state.w)
        self.t.append(t)

class TargetCourse:
    """Path (cx, cy) and look-ahead parameters"""
    def __init__(self, cx, cy, k=1.25, Lfc=1.5):
        self.cx = cx
        self.cy = cy
        self.k = k
        self.Lfc = Lfc
        self.old_nearest_point_index = None

    def search_target_index(self, state: DiffState):
        # find nearest index
        if self.old_nearest_point_index is None:
            dx = [state.x - icx for icx in self.cx]
            dy = [state.y - icy for icy in self.cy]
            d = np.hypot(dx, dy)
            ind = int(np.argmin(d))
            self.old_nearest_point_index = ind
        else:
            ind = self.old_nearest_point_index
            while True:
                if ind + 1 >= len(self.cx): break
                d_curr = math.hypot(state.x - self.cx[ind], state.y - self.cy[ind])
                d_next = math.hypot(state.x - self.cx[ind+1], state.y - self.cy[ind+1])
                if d_curr < d_next: break
                ind += 1
            self.old_nearest_point_index = ind
        # look-ahead distance
        Lf = self.k * state.v + self.Lfc
        # advance to point beyond look-ahead
        while Lf > math.hypot(state.x - self.cx[ind], state.y - self.cy[ind]):
            if ind + 1 >= len(self.cx): break
            ind += 1
        return ind, Lf

class PurePursuitController:
    """
    Encapsulates PD pure-pursuit control for differential drive using v, w.
    """
    def __init__(self, k=1.25, Lfc=1.5,
                 Kp_speed=2.4, Kp_heading=1.6, Kd_heading=0.8,
                 dt=0.1):
        self.k = k
        self.Lfc = Lfc
        self.Kp_speed = Kp_speed
        self.Kp_heading = Kp_heading
        self.Kd_heading = Kd_heading
        self.pp_dt = dt
        self.prev_alpha = 0.0

    def control(self, state: DiffState, trajectory: TargetCourse,
                prev_ind, target_speed):
        ind, Lf = trajectory.search_target_index(state)
        if prev_ind >= ind:
            ind = prev_ind
        # goal position
        if ind < len(trajectory.cx):
            tx, ty = trajectory.cx[ind], trajectory.cy[ind]
        else:
            tx, ty = trajectory.cx[-1], trajectory.cy[-1]
            ind = len(trajectory.cx) - 1
        # heading error
        alpha = math.atan2(ty - state.y, tx - state.x) - state.yaw
        alpha = (alpha + math.pi) % (2*math.pi) - math.pi
        # pure pursuit curvature
        curvature = 2.0 * math.sin(alpha) / Lf
        # derivative of heading error
        dalpha = (alpha - self.prev_alpha) / self.pp_dt
        # PD angular velocity command
        w_cmd = self.Kp_heading * curvature * state.v + self.Kd_heading * dalpha
        # P speed control for linear accel
        accel = self.Kp_speed * (target_speed - state.v)
        v_cmd = state.v + accel * self.pp_dt
        self.prev_alpha = alpha
        return v_cmd, w_cmd, ind

class PurePursuitNode(Node):
    """ROS2 node publishing Twist messages from pure pursuit controller."""
    def __init__(self):
        super().__init__('pure_pursuit_node')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        # parameters
        self.pp_dt = 0.1
        self.target_speed = 10.0 / 3.6  # [m/s]
        # path setup
        cx = np.arange(0, 50, 0.5)
        cy = [math.sin(ix / 5.0) * ix / 2.0 for ix in cx]
        self.state_now = DiffState(x=0.0, y=-3.0, yaw=0.0, v=0.0, w=0.0, dt=self.pp_dt)
        self.controller = PurePursuitController(dt=self.pp_dt)
        self.trajectory = TargetCourse(cx, cy)
        self.prev_ind = 0
        self.time = 0.0
        self.states = States()
        self.states.append(self.time, self.state_now)

    def timer_callback(self):
        # compute control
        v_cmd, w_cmd, self.prev_ind = self.controller.control(
            self.state_now, self.trajectory, self.prev_ind, self.target_speed)
        # update state
        self.state_now.update(v_cmd, w_cmd)
        self.time += self.pp_dt
        self.states.append(self.time, self.state_now)
        # publish Twist
        twist = Twist()
        twist.linear.x = v_cmd
        twist.angular.z = w_cmd
        self.publisher_.publish(twist)
        # optional visualization
        if show_animation:
            plt.cla()
            plt.arrow(self.state_now.x, self.state_now.y,
                      math.cos(self.state_now.yaw), math.sin(self.state_now.yaw),
                      head_width=0.3, head_length=0.3)
            plt.plot(self.trajectory.cx, self.trajectory.cy, '--r')
            plt.plot(self.states.x, self.states.y, '-b')
            plt.axis('equal'); plt.grid(True)
            plt.pause(0.001)

def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    # final plot
    if show_animation:
        plt.cla()
        plt.plot(node.trajectory.cx, node.trajectory.cy, '.r')
        plt.plot(node.states.x, node.states.y, '-b')
        plt.axis('equal'); plt.grid(True)
        plt.figure()
        plt.plot(node.states.t, [v*3.6 for v in node.states.v], '-r')
        plt.xlabel('Time [s]'); plt.ylabel('Speed [km/h]'); plt.grid(True)
        plt.show()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
