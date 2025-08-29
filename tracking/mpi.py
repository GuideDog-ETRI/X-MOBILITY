"""
Unicycle Robot을 위한 향상된 경로 추적 알고리즘 (MPPI)

주요 개선사항:
1. 헤딩 오차를 고려한 비용 함수
2. 횡방향/종방향 오차 분리
3. 곡률 기반 속도 제어
4. 적응적 Look-ahead 거리
5. 경로 preview 기반 제어
6. Unicycle 운동학 모델 적용 (제어 입력: v, w)
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import copy
from rclpy.node import Node
from geometry_msgs.msg import Twist

show_animation = True

class State:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, w=0.0, dt=0.1):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.w = w
        self.dt = dt

    def update(self, v_cmd, w_cmd):
        """Unicycle 운동학 모델 업데이트"""
        # 속도 제한
        # self.v = max(-15.0, min(v_cmd, 15.0))
        # self.w = max(-10.0, min(w_cmd, 10.0))  # 각속도 제한
        self.v = v_cmd
        self.w = w_cmd
        
        # 상태 업데이트
        self.x += self.v * math.cos(self.yaw) * self.dt
        self.y += self.v * math.sin(self.yaw) * self.dt
        self.yaw += self.w * self.dt
        
        # yaw 각도 정규화
        self.yaw = self.normalize_angle(self.yaw)

    def normalize_angle(self, angle):
        """각도를 [-π, π] 범위로 정규화"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def calc_distance(self, point_x, point_y):
        dx = self.x - point_x
        dy = self.y - point_y
        return math.hypot(dx, dy)
    
    def copy(self):
        return State(self.x, self.y, self.yaw, self.v, self.w, self.dt)


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
    def __init__(self, cx, cy, k=1.00, Lfc=2.0):
        self.cx = cx
        self.cy = cy
        self.k = k
        self.Lfc = Lfc
        self.old_nearest_point_index = None
        self.curvatures = self.calc_curvatures()
        self.headings = self.calc_headings()

    def calc_curvatures(self):
        """중앙차분 기반의 곡률 계산"""
        curvatures = [0.0]
        for i in range(1, len(self.cx) - 1):
            x0, x1, x2 = self.cx[i - 1], self.cx[i], self.cx[i + 1]
            y0, y1, y2 = self.cy[i - 1], self.cy[i], self.cy[i + 1]

            dx1 = x1 - x0
            dx2 = x2 - x1
            dy1 = y1 - y0
            dy2 = y2 - y1

            ddx = dx2 - dx1
            ddy = dy2 - dy1

            denominator = (dx1**2 + dy1**2)**1.5
            numerator = dx1 * ddy - dy1 * ddx

            if denominator == 0:
                curvature = 0.0
            else:
                curvature = abs(numerator / denominator)

            curvatures.append(curvature)

        curvatures.append(0.0)  # 마지막 점
        return curvatures

    def calc_headings(self):
        """경로의 헤딩 각도 계산"""
        headings = []
        for i in range(len(self.cx)):
            if i == len(self.cx) - 1:
                headings.append(headings[-1])
            else:
                dx = self.cx[i+1] - self.cx[i]
                dy = self.cy[i+1] - self.cy[i]
                heading = math.atan2(dy, dx)
                headings.append(heading)
        return headings

    def search_target_index(self, state):
        # 1. 가장 가까운 경로 인덱스 찾기
        if self.old_nearest_point_index is None:
            dx = [state.x - icx for icx in self.cx]
            dy = [state.y - icy for icy in self.cy]
            d = np.hypot(dx, dy)
            ind = np.argmin(d)
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

        # 2. 곡률 기반 적응적 Look-ahead 거리 계산
        curvature = self.curvatures[min(ind, len(self.curvatures) - 1)]
        dcurvatures = np.gradient(self.curvatures)
        dcurv = abs(dcurvatures[min(ind, len(dcurvatures) - 1)])
        curvature_factor = 1.0 / (1.0 + 10.0 * curvature + 5.0 * dcurv)
        lookahead_len = 5
        curv_samples = self.curvatures[ind:ind+lookahead_len]
        curv_samples = curv_samples if len(curv_samples) > 0 else [0.0]
        curv_mean = np.mean(curv_samples)

        curvature_factor = 1.0 / (1.0 + 10.0 * curv_mean)
        if curvature_factor == 0:  # 직선 구간
            Lf = max(self.Lfc, 1.0)  # 직선에서는 Lf를 고정
        else:  # 곡선 구간
            Lf = max(self.k * state.v * curvature_factor + self.Lfc, 1.0)


        # 3. 누적 경로 길이 기준으로 목표 인덱스 계산
        target_ind = ind
        accumulated_distance = state.calc_distance(self.cx[ind], self.cy[ind])  # 보정 추가

        while accumulated_distance < Lf and (target_ind + 1) < len(self.cx):
            dx = self.cx[target_ind + 1] - self.cx[target_ind]
            dy = self.cy[target_ind + 1] - self.cy[target_ind]
            segment_dist = math.hypot(dx, dy)
            accumulated_distance += segment_dist
            target_ind += 1


        # 4. Look-ahead 내에서 곡률 변화율이 큰 구간 우선 반영 (Lf 이내만)
        lookahead_range = range(ind, target_ind + 1)
        max_dcurv = 0.0
        max_dcurv_ind = target_ind
        for i in lookahead_range:
            if i >= len(dcurvatures):
                break
            if abs(dcurvatures[i]) > max_dcurv:
                max_dcurv = abs(dcurvatures[i])
                max_dcurv_ind = i

        dcurv_threshold = 0.05
        if max_dcurv > dcurv_threshold:
            target_ind = max_dcurv_ind

        return target_ind, Lf
    
    def calc_cross_track_error(self, state, nearest_ind):
        """횡방향 오차 계산"""
        if nearest_ind >= len(self.cx) - 1:
            nearest_ind = len(self.cx) - 2

        # 가장 가까운 경로 세그먼트 찾기
        p1x, p1y = self.cx[nearest_ind], self.cy[nearest_ind]
        p2x, p2y = self.cx[nearest_ind + 1], self.cy[nearest_ind + 1]

        # 점-직선 거리 계산 계수
        A = p2y - p1y
        B = p1x - p2x
        C = p2x * p1y - p1x * p2y

        denom = math.hypot(A, B)

        if denom == 0:
            # 두 점이 같은 경우: 세그먼트가 성립하지 않음
            cross_track_error = 0.0  # 또는 예외처리 raise ValueError(...) 가능
        else:
            cross_track_error = abs(A * state.x + B * state.y + C) / denom

            # 오차 방향 결정 (왼쪽: 양수, 오른쪽: 음수)
            cross_product = (p2x - p1x) * (state.y - p1y) - (p2y - p1y) * (state.x - p1x)
            if cross_product < 0:
                cross_track_error *= -1

        return cross_track_error



def mppi_control(state, target_course, target_ind, target_speed, N=120, horizon=12, dt=0.1):
    """
    Unicycle Robot을 위한 향상된 MPPI 제어 
    제어 입력: (v, w) - 선속도, 각속도
    """
    # 목표 지점 계산
    target_x = target_course.cx[min(target_ind, len(target_course.cx) - 1)]
    target_y = target_course.cy[min(target_ind, len(target_course.cy) - 1)]
    
    # 목표 방향으로의 각속도 계산
    dx = target_x - state.x
    dy = target_y - state.y
    
    # dx, dy가 0일 때 특별 처리
    if dx == 0 and dy == 0:
        target_angle = state.yaw  # 방향을 현재 yaw와 같게 설정
    else:
        target_angle = math.atan2(dy, dx)  # 일반적인 경우에는 atan2 사용
    
    angle_diff = target_angle - state.yaw
    
    # 목표 도달 여부 체크
    last_ind = len(target_course.cx) - 1
    goal_x = target_course.cx[last_ind]
    goal_y = target_course.cy[last_ind]
    distance_to_goal = state.calc_distance(goal_x, goal_y)
    goal_reached = distance_to_goal < 1.0
    
    # 목표 각속도 계산
    w_mean = 0.0 if goal_reached else angle_diff * 2.0

    # 각도 정규화
    while angle_diff > math.pi:
        angle_diff -= 2 * math.pi
    while angle_diff < -math.pi:
        angle_diff += 2 * math.pi
    
    # 적응적 샘플링
    v_mean = target_speed
    w_mean = angle_diff * 2.0  # 목표 방향으로 편향
    v_std = 0.1
    #curvature = target_course.curvatures[min(target_ind, len(target_course.curvatures)-1)]

    if target_ind >= len(target_course.curvatures):
        curvature = 0.0  # 곡률 0 (직선)
    else:
        curvature = target_course.curvatures[target_ind]
    
    w_std = min(1.0, 0.3 + 3.0 * curvature)
    v_samples = np.random.normal(v_mean, v_std, size=(N, horizon))
    w_samples = np.random.normal(w_mean, w_std, size=(N, horizon))
    
    def smooth_clip(x, limit):
        # limit이 0일 때 NaN을 방지
        if limit == 0:
            print(f"Warning: limit is 0, returning x={x} without clipping")
            return x  # limit이 0이면 그냥 x를 반환하는 방식으로 처리
        clipped_value = limit * np.tanh(x / limit)
        return clipped_value
    
    # 제어 입력 제한
    v_samples = np.array([smooth_clip(v, 8.0) for v in v_samples.flatten()]).reshape(v_samples.shape)
    w_samples = np.array([smooth_clip(w, 2.5) for w in w_samples.flatten()]).reshape(w_samples.shape)

    costs = np.zeros(N)

    for i in range(N):
        sim_state = state.copy()
        cost = 0.0

        for t in range(horizon):
            v_cmd = v_samples[i, t]
            w_cmd = w_samples[i, t]
            sim_state.update(v_cmd, w_cmd)

            # 현재 시뮬레이션 상태에서 가장 가까운 경로 지점 찾기
            cx_arr = np.array(target_course.cx)
            cy_arr = np.array(target_course.cy)
            dx = sim_state.x - cx_arr
            dy = sim_state.y - cy_arr
            distances = np.hypot(dx, dy)
            nearest_ind = np.argmin(distances)
            
            # 목표 지점 계산
            sim_target_ind, _ = target_course.search_target_index(sim_state)
            sim_target_ind = min(sim_target_ind, len(target_course.cx) - 1)
            
            target_x = target_course.cx[sim_target_ind]
            target_y = target_course.cy[sim_target_ind]
            target_heading = target_course.headings[sim_target_ind]
            
            # 1. 횡방향 오차 (Cross-track error)
            cross_track_error = target_course.calc_cross_track_error(sim_state, nearest_ind)
            
            # 2. 헤딩 오차
            heading_error = sim_state.yaw - target_heading
            heading_error = sim_state.normalize_angle(heading_error)
            
            # 3. 속도 오차
            curvature = target_course.curvatures[sim_target_ind]
            speed_limit = min(target_speed, max(0.5, 4.0 / (1.0 + 15.0 * curvature)))
            speed_error = abs(sim_state.v - speed_limit)
            
            # 4. 진행 방향 오차 (목표 지점까지의 거리)
            distance_to_target = sim_state.calc_distance(target_x, target_y)
            
            # 5. 제어 입력 페널티 (unicycle에 맞게 수정)
            v_penalty = 0.01 * ((v_cmd - speed_limit)**2)
            w_penalty = 0.05 * (w_cmd**2)
            control_penalty = v_penalty + w_penalty
            
            # 6. 급격한 방향 변화 페널티
            angular_acceleration = abs(w_cmd - sim_state.w)
            angular_penalty = 0.9 * angular_acceleration**2 if not goal_reached else 0.0
            
            # 7. 진행 보상
            progress_reward = -0.2 * sim_state.v if sim_state.v > 0 else 0.0
            
            # 8. 경로 완주 보상
            completion_reward = 0.0
            if sim_target_ind > len(target_course.cx) * 0.8:
                completion_reward = -1.0
            
            # 가중치 적용한 총 비용
            discount = 0.95 ** t
            # pathalign 추가: 코사인 값 (0~1)
            pathalign = (1 + math.cos(heading_error)) / 2  # 0~1, 1이 완벽 정렬

            # 비용에 반영 (heading error 항목 대체하거나 별도로 추가)
            # pathalign이 낮으면 비용 증가
            pathalign_cost = 10.0 * (1 - pathalign)**2 if not goal_reached else 0.0 # 가중치 5.0 적용
            curvature = target_course.curvatures[sim_target_ind]
            cte_weight = 30.0 * (curvature + 0.05)  # 곡률이 작으면 작은 가중치
            # 기존 heading error 비용 대신 pathalign_cost 사용
            step_cost = (
                cte_weight * cross_track_error**2 +           # 횡방향 오차 (가장 중요)
                pathalign_cost +                       # pathalign 비용
                1.0 * speed_error**2 +                 # 속도 오차
                0.3 * distance_to_target**2 +          # 거리 오차
                control_penalty +                      # 제어 입력 페널티
                angular_penalty +                      # 각속도 변화 페널티
                progress_reward +                      # 진행 보상
                completion_reward                      # 완주 보상
            )

            cost += discount * step_cost

        costs[i] = cost

    # 가중치 계산 및 제어 입력 선택
    lambda_ = 1.0  # 더 작은 lambda로 exploitation 증가
    min_cost = np.min(costs)
    costs_normalized = costs - min_cost
    
    weights = np.exp(-costs_normalized / lambda_)
    weights = np.clip(weights, 1e-10, None)
    weights /= np.sum(weights)

    v_mppi = np.sum(weights * v_samples[:, 0])
    w_mppi = np.sum(weights * w_samples[:, 0])
    
    # 부드러운 제어를 위한 필터링
    alpha = 0.5
    v_mppi = alpha * v_mppi + (1 - alpha) * state.v
    w_mppi = alpha * w_mppi + (1 - alpha) * state.w
    
    return v_mppi, w_mppi

class MPPINode(Node):
    """ROS2 node publishing Twist messages using MPPI controller."""
    def __init__(self):
        super().__init__('mppi_node')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        # parameters
        self.dt = 0.1
        self.target_speed = 2.34 / 3.6  # [m/s]
        
        # path setup
        cx = np.arange(0, 50, 0.5)
        cy = [math.sin(ix / 5.0) * ix / 2.0 for ix in cx]
        
        self.state_now = State(x=0.0, y=-3.0, yaw=0.0, v=0.0, w=0.0, dt=self.dt)
        #self.controller= mppi_control()
        self.trajectory = TargetCourse(cx, cy)
        self.time = 0.0
        self.states = States()
        self.states.append(self.time, self.state_now)

    def timer_callback(self):
        # find target index on the path
        target_ind, _ = self.trajectory.search_target_index(self.state_now)

        # compute control using MPPI
        v_cmd, w_cmd = mppi_control(self.state_now, self.trajectory, target_ind, self.target_speed)

        # update state
        self.state_now.update(v_cmd, w_cmd)
        self.time += self.dt
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


def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    if not isinstance(x, float):
        for ix, iy, iyaw in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)


def main():
    # 파라미터 설정
    k = 1.0  # look forward gain
    Lfc = 2.0  # look-ahead distance
    dt = 0.1
    
    # 더 도전적인 경로 생성
    cx = np.linspace(0, 60, 300)  # x좌표
    cy = np.sin(cx / 3) * np.cos(cx / 8) * 4  # x에 따라 y를 결정

    target_speed = 0.65  # [m/s] - 더 낮은 속도로 안정성 증가
    T = 150.0

    # 초기 상태
    state = State(x=0.0, y=-2.0, yaw=0.0, v=0.0, w=0.0, dt=dt)

    lastIndex = len(cx) - 1
    time = 0.0
    states = States()
    states.append(time, state)
    target_course = TargetCourse(cx=cx, cy=cy, k=k, Lfc=Lfc)
    target_ind, _ = target_course.search_target_index(state)

    # 성능 지표
    cross_track_errors = []
    
    while T >= time and target_ind < lastIndex - 5:
        # 현재 타겟 인덱스 업데이트
        target_ind, _ = target_course.search_target_index(state)
        
        # 성능 지표 계산
        nearest_distances = [state.calc_distance(x, y) for x, y in zip(cx, cy)]
        nearest_ind = np.argmin(nearest_distances)
        cross_track_error = target_course.calc_cross_track_error(state, nearest_ind)
        cross_track_errors.append(abs(cross_track_error))
        
        # MPPI 제어
        v_cmd, w_cmd = mppi_control(
            state, target_course, target_ind, target_speed, 
            N=200, horizon=5, dt=dt
        )

        state.update(v_cmd, w_cmd)
        time += dt
        states.append(time, state)

        # 목표 도달 확인
        distance_to_goal = state.calc_distance(cx[lastIndex], cy[lastIndex])
        if distance_to_goal < 1.0:
            print(f"목표 지점 도달! 거리: {distance_to_goal:.2f}m")
            
            # 목표점에 도달하면 현재 위치로 고정
            state.x = cx[lastIndex]
            state.y = cy[lastIndex]
            state.v = 0.0
            state.w = 0.0
            break

        if show_animation:
            plt.cla()
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plot_arrow(state.x, state.y, state.yaw)
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(states.x, states.y, "-b", label="trajectory")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            plt.axis("equal")
            plt.grid(True)
            plt.title(f"v: {state.v:.1f}m/s, w: {state.w:.2f}rad/s, Cross-track error: {cross_track_error:.2f}m")
            plt.pause(0.001)

    # 성능 평가
    avg_cross_track_error = np.mean(cross_track_errors)
    max_cross_track_error = np.max(cross_track_errors)
    print(f"평균 횡방향 오차: {avg_cross_track_error:.3f}m")
    print(f"최대 횡방향 오차: {max_cross_track_error:.3f}m")
    print(f"최종 목표 인덱스: {target_ind}/{lastIndex}")

    if show_animation:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(cx, cy, ".r", label="course")
        plt.plot(states.x, states.y, "-b", label="trajectory")
        plt.legend()
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.axis("equal")
        plt.grid(True)
        plt.title("Unicycle Robot Path Tracking Result")

        plt.subplot(1, 2, 2)
        plt.plot(states.t, states.v, "-r", label="Linear velocity [m/s]")
        plt.plot(states.t, states.w, "-g", label="Angular velocity [rad/s]")
        plt.plot(states.t[:len(cross_track_errors)], cross_track_errors, "-b", label="Cross-track error [m]")
        plt.xlabel("Time[s]")
        plt.ylabel("Control inputs / Error")
        plt.legend()
        plt.grid(True)
        plt.title("Control Performance")
        
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    print("Unicycle Robot MPPI 경로 추적 시뮬레이션")
    main()