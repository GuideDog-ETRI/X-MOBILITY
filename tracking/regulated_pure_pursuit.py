"""

Path tracking simulation with pure pursuit steering and PID speed control.

author: Atsushi Sakai (@Atsushi_twi)
        Guillaume Jacquenot (@Gjacquenot)

"""
import numpy as np
import math
import matplotlib.pyplot as plt
# from ipdb import set_trace as bp

'''
# Parameters
k = 1.25  # look forward gain
Lfc = 1.5  # default 0.3[m] look-ahead distance
Kp = 2.4 # 1.6  # speed proportional gain
Ka = 1.6 # angular velocity propotional gain 
dt = 0.1  # [s] time tick
WB = 0.000001  # [m] wheel base of vehicle (jackal 0.26), Distance between front-wheel and rear-whell
'''
show_animation = True

class State:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, w=0.0, dt=0.1, WB=0.26):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.w = w
        self.dt = dt
        self.WB = WB

    def update(self, a, delta):
        self.x += self.v * math.cos(self.yaw) * self.dt
        self.y += self.v * math.sin(self.yaw) * self.dt
        self.yaw += self.v / self.WB * math.tan(delta) * self.dt
        self.v += a * self.dt
        self.w = self.v * math.tan(delta) / self.WB
        ''' w = v/R, R(:회전반경) = WB(:휠베이스) / tan(조향각:delta),
        ==> w = v * tan(delta) / WB '''

    def calc_distance(self, point_x, point_y):
        dx = self.x - point_x
        dy = self.y - point_y
        return math.hypot(dx, dy)


class States:
    def __init__(self):
        self.x = []
        self.y = []
        self.yaw = []
        self.v = []
        self.w = []
        self.t = []

    def append(self, t, state):
        self.x.append(state.x)
        self.y.append(state.y)
        self.yaw.append(state.yaw)
        self.v.append(state.v)
        self.w.append(state.w)
        self.t.append(t)


class TargetCourse:
    def __init__(self, cx, cy, k=1.25, Lfc=1.5):
        self.cx = cx
        self.cy = cy
        self.k = k
        self.Lfc = Lfc
        self.old_nearest_point_index = None

    def search_target_index(self, state):
        if self.old_nearest_point_index is None:
            dx = [state.x - icx for icx in self.cx]
            dy = [state.y - icy for icy in self.cy]
            d = np.hypot(dx, dy)
            ind = np.argmin(d)
            self.old_nearest_point_index = ind
        else:
            ind = self.old_nearest_point_index
            distance_this_index = state.calc_distance(self.cx[ind], self.cy[ind])

            while True:
                if (ind + 1) >= len(self.cx):
                    break
                distance_next_index = state.calc_distance(self.cx[ind + 1], self.cy[ind + 1])
                if distance_this_index < distance_next_index:
                    break
                ind += 1
                distance_this_index = distance_next_index
            self.old_nearest_point_index = ind

        Lf = self.k * state.v + self.Lfc

        while Lf > state.calc_distance(self.cx[ind], self.cy[ind]):
            if (ind + 1) >= len(self.cx):
                break
            ind += 1
        return ind, Lf


def proportional_control(target, current, Kp=2.4):
    return Kp * (target - current)


def pure_pursuit_steer_control(state, trajectory, pind, Kp=2.4, Ka=1.6, dt=0.1, WB=0.26, regulating_Ka=1.0):
    ind, Lf = trajectory.search_target_index(state)

    if pind >= ind:
        ind = pind

    if ind < len(trajectory.cx):
        tx = trajectory.cx[ind]
        ty = trajectory.cy[ind]
    else:  # toward goal
        tx = trajectory.cx[-1]
        ty = trajectory.cy[-1]
        ind = len(trajectory.cx) - 1

    alpha = math.atan2(ty - state.y, tx - state.x) - state.yaw

    delta = math.atan2(2.0 * WB * math.sin(alpha) / Lf, 1.0) * Ka

    # regulate_steering 함수로 조향각을 조정
    delta = regulate_steering(delta, state.calc_distance(tx, ty), regulating_Ka)    

    # 조향각을 -π에서 π 사이의 값으로 맞춤 (normalize) (실험중, 추가)
    #delta = (delta + math.pi) % (2 * math.pi) - math.pi    

    return delta, ind


def _search_optimal_regulate_Ka(steering_angle=0.785, distance=0.1):
    # Constants
    #steering_angle = 3.14/4  # Steering angle in rad
    #regulating_Ka = 0.1  # Regulating factor
    regulating_Ka = np.linspace(0, 100, 100)  # Distance to target from 0 to 1 meters
    
    # Compute regulated angle using regulate_steering function
    regulated_angle = [regulate_steering(steering_angle, distance, Ka) for Ka in regulating_Ka]
    regulated_angle_ratio = [regulate_steering(steering_angle/steering_angle, distance, Ka) for Ka in regulating_Ka]

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.plot(regulating_Ka, regulated_angle_ratio, label="Regulated Angle Ratio", color="b")
    plt.title(f"Regulated Angle vs rKa at {distance}m.")
    plt.xlabel("regulating_Ka")
    plt.ylabel("Regulated Angle Scaling Factor")
    plt.grid(True)
    plt.legend()
    plt.show()    

    for i in range(len(regulated_angle_ratio)):
        print(f"rKa:regulated_angle_ratio = {regulating_Ka[i]:.1f}:{regulated_angle_ratio[i]:.1f}")

    '''
    rKa:regulated_angle_ratio = 0.0:0.5
    rKa:regulated_angle_ratio = 1.0:0.5
    rKa:regulated_angle_ratio = 2.0:0.6
    rKa:regulated_angle_ratio = 3.0:0.6
    rKa:regulated_angle_ratio = 4.0:0.6
    rKa:regulated_angle_ratio = 5.1:0.6
    rKa:regulated_angle_ratio = 6.1:0.6
    rKa:regulated_angle_ratio = 7.1:0.7
    rKa:regulated_angle_ratio = 8.1:0.7
    rKa:regulated_angle_ratio = 9.1:0.7
    rKa:regulated_angle_ratio = 10.1:0.7
    rKa:regulated_angle_ratio = 11.1:0.8
    rKa:regulated_angle_ratio = 12.1:0.8
    rKa:regulated_angle_ratio = 13.1:0.8
    rKa:regulated_angle_ratio = 14.1:0.8
    rKa:regulated_angle_ratio = 15.2:0.8
    rKa:regulated_angle_ratio = 16.2:0.8
    rKa:regulated_angle_ratio = 17.2:0.8
    rKa:regulated_angle_ratio = 18.2:0.9
    rKa:regulated_angle_ratio = 19.2:0.9
    rKa:regulated_angle_ratio = 20.2:0.9
    rKa:regulated_angle_ratio = 21.2:0.9
    rKa:regulated_angle_ratio = 22.2:0.9
    rKa:regulated_angle_ratio = 23.2:0.9
    rKa:regulated_angle_ratio = 24.2:0.9
    rKa:regulated_angle_ratio = 25.3:0.9
    rKa:regulated_angle_ratio = 26.3:0.9
    rKa:regulated_angle_ratio = 27.3:0.9
    rKa:regulated_angle_ratio = 28.3:0.9
    rKa:regulated_angle_ratio = 29.3:0.9
    rKa:regulated_angle_ratio = 30.3:1.0
    '''

def _plot_regulate_graph(regulating_Ka=1.0):
    # Constants
    steering_angle = 3.14/4  # Steering angle in rad
    #regulating_Ka = 0.1  # Regulating factor
    distance_to_target = np.linspace(0, 1, 100)  # Distance to target from 0 to 1 meters
    
    # Compute regulated angle using regulate_steering function
    regulated_angle = [regulate_steering(steering_angle, dist, regulating_Ka) for dist in distance_to_target]
    
    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.plot(distance_to_target, regulated_angle, label="Regulated Angle", color="b")
    plt.title("Regulated Angle vs Distance to Target")
    plt.xlabel("Distance to Target (m)")
    plt.ylabel("Regulated Angle (Rad)")
    plt.grid(True)
    plt.legend()
    plt.show()    

    
def regulate_steering(steering_angle, distance_to_target, regulating_Ka=1.0):
    # 거리 비례로 조향각 조정 (regulating_Ka는 조정 비율, sigma value)
    #regulated_angle = steering_angle / (1 + regulating_Ka * distance_to_target)
    regulated_angle = steering_angle / (1 + math.exp(-regulating_Ka * distance_to_target))
    '''  For steering_angle = 0.785(45degree), out_rad at 0.1 meter distance
            In case of regulating_Ka=1000.0 :  0.785
            In case of regulating_Ka= 100.0 :  0.785
            In case of regulating_Ka=  10.0 :  0.57
            In case of regulating_Ka=   1.0 :  0.39
            In case of regulating_Ka=0.0001 :  0.39

    '''
    return regulated_angle


def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    """
    Plot arrow
    """

    if not isinstance(x, float):
        for ix, iy, iyaw in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)


def main(): # simulation
    # Parameters
    k = 1.25  # look forward gain
    Lfc = 1.5  # default 0.3[m] look-ahead distance
    Kp = 2.4 # 1.6  # speed proportional gain
    Ka = 1.6 # angular velocity propotional gain 
    dt = 0.1  # [s] time tick
    WB = 0.000001  # [m] wheel base of vehicle (jackal 0.26), Distance between front-wheel and rear-whell

    #  target course
    cx = np.arange(0, 50, 0.5)
    cy = [math.sin(ix / 5.0) * ix / 2.0 for ix in cx]

    target_speed = 10.0 / 3.6  # [m/s]

    T = 100.0  # max simulation time

    # initial state
    state = State(x=-0.0, y=-3.0, yaw=0.0, v=0.0, dt=dt, WB=WB)

    lastIndex = len(cx) - 1
    time = 0.0
    states = States()
    states.append(time, state)
    target_course = TargetCourse(cx=cx, cy=cy, k=k, Lfc=Lfc)
    target_ind, _ = target_course.search_target_index(state)

    while T >= time and lastIndex > target_ind:
        # Calc control input
        ai = proportional_control(target_speed, state.v)
        di, target_ind = pure_pursuit_steer_control(
            state, target_course, target_ind, Kp, Ka, dt, WB)

        state.update(ai, di)  # Control vehicle

        time += dt
        states.append(time, state)

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plot_arrow(state.x, state.y, state.yaw)
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(states.x, states.y, "-b", label="trajectory")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            plt.axis("equal")
            plt.grid(True)
            plt.title("Speed[km/h]:" + str(state.v * 3.6)[:4])
            plt.pause(0.001)

    # Test
    assert lastIndex >= target_ind, "Cannot goal"

    if show_animation:  # pragma: no cover
        plt.cla()
        plt.plot(cx, cy, ".r", label="course")
        plt.plot(states.x, states.y, "-b", label="trajectory")
        plt.legend()
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.axis("equal")
        plt.grid(True)

        plt.subplots(1)
        plt.plot(states.t, [iv * 3.6 for iv in states.v], "-r")
        plt.xlabel("Time[s]")
        plt.ylabel("Speed[km/h]")
        plt.grid(True)
        plt.show()




if __name__ == '__main__':
    print("Pure pursuit path tracking simulation start")
    _search_optimal_regulate_Ka()
    #main()
