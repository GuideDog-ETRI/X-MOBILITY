import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from gdm_planning.angle import rot_mat_2d
from gdm_planning.grid_map import GridMap
from collections import OrderedDict
from itertools import cycle

import heapq
import math
import numpy as np

class QuinticPolynomial: # QPP 방법을 이용하여 경로 부드럽게 생성
    def __init__(self, xs, vxs, axs, xe, vxe, axe, T): # xs: 시작위치 / vxs: 시작속도 / axs: 시작가속도 / xe: 끝위치 / vxe: 끝속도 / axe: 끝가속도
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([ # a3, a4, a5 계수를 풀기 위해 만든 선형 방정식 계수
            [T**3,   T**4,    T**5],
            [3*T**2, 4*T**3,  5*T**4],
            [6*T,    12*T**2, 20*T**3]
        ])
        B = np.array([
            xe - self.a0 - self.a1 * T - self.a2 * T**2,
            vxe - self.a1 - 2 * self.a2 * T,
            axe - 2 * self.a2
        ])
        x = np.linalg.solve(A, B)
        self.a3, self.a4, self.a5 = x

    def calc_point(self, t): # 시간 t에서의 위치값 계산
        return (self.a0 + self.a1 * t + self.a2 * t**2 + self.a3 * t**3 +
                self.a4 * t**4 + self.a5 * t**5)
    
class QuinticTrajectory2D: # QuinticPolynomial에서 만들어준 x축 이동용 다항식, y축 이동용 다항식을 합쳐서 2D 평면상의 부드러운 궤적을 만들어주는 클래스
    def __init__(self, start, end, T):
        self.qp_x = QuinticPolynomial(start[0], 0, 0, end[0], 0, 0, T)
        self.qp_y = QuinticPolynomial(start[1], 0, 0, end[1], 0, 0, T)

    def calc_point(self, t): # 2D 좌표 리스트로 반환
        return [self.qp_x.calc_point(t), self.qp_y.calc_point(t)]

def smooth_path_quintic(path, interval=0.05):
    if len(path) < 2:
        return path

    smoothed = []
    for i in range(len(path) - 1):
        start = path[i]
        end = path[i + 1]
        T = math.hypot(end[0] - start[0], end[1] - start[1])
        if T < 1e-6:
            continue

        traj = QuinticTrajectory2D(start, end, T)

        t = 0.0
        while t < T:
            smoothed.append(traj.calc_point(t))
            t += interval
    smoothed.append(path[-1])
    return smoothed

class YUNJUplanner:
    """
    A* path planner for grid-based maps.
    Uses an 8-connected grid and Euclidean heuristic for optimal pathfinding on a discretized occupancy grid.
    """
    def __init__(self, start, goal, grid_map):
        """
        :param grid_map: GridMap instance with attributes:
                         - width, height: dimensions in cells
                         - resolution: cell size in meters
                         - center_x, center_y: world coordinates of grid center
                         - grid_map.check_occupancy_from_xy_index(x_idx, y_idx): returns True if occupied
                         - grid_map.get_xy_index_from_xy_pos(x, y): returns (x_idx, y_idx, valid)
        """
        self.grid_map = grid_map
        self.width = grid_map.width # 격자 지도의 가로 셀 개수
        self.height = grid_map.height # 격자 지도의 세로 셀 개수
        self.resolution = grid_map.resolution # 셀 하나의 크기
        self.center_x = grid_map.center_x # 지도의 중심 좌표(X)
        self.center_y = grid_map.center_y # 지도의 중심 좌표(Y)
        # Compute world coordinates of lower-left grid corner
        self.origin_x = self.center_x - (self.width * self.resolution) / 2.0 # 지도의 좌하단 X좌표
        self.origin_y = self.center_y - (self.height * self.resolution) / 2.0 # 지도의 좌하단 Y좌표

        self.start = start
        self.goal = goal

        self.distance_cache = OrderedDict() # 장애물과의 거리를 계산한 결과를 캐싱하여 중복 방지(LRU방식 사용(가장 오랫동안 사용되지 않은 데이터를 우선적으로 제거하는 캐시 알고리즘))
        self.distance_cache_max_size = 10000

        self.obstacle_search_radius = 7
        self.distance_sorted_deltas = sorted(
            [(dx, dy, math.hypot(dx, dy)) for dx in range(-self.obstacle_search_radius, self.obstacle_search_radius + 1)
            for dy in range(-self.obstacle_search_radius, self.obstacle_search_radius + 1)],
            key=lambda x: x[2]
        )

    def plan(self, animation=False):
        """
        Plans a path from start to goal.
        :param start: [x, y] in world coordinates
        :param goal: [x, y] in world coordinates
        :return: List of [x, y] waypoints in world coordinates or None if no path.
        """
        # Convert to grid indices / 시작점, 목표점을 격자 인덱스로 변환
        sx, sy, valid_s = self.grid_map.get_xy_index_from_xy_pos(self.start[0], self.start[1]) # 월드 좌표를 격자 인덱스로 반환하며, 반환 가능 여부도 함께 반환
        gx, gy, valid_g = self.grid_map.get_xy_index_from_xy_pos(self.goal[0], self.goal[1])
        if not valid_s or not valid_g: # 격자 내에 존재하지 않는다면 경로 탐색 불가
            return None
        start_idx = (sx, sy)
        goal_idx = (gx, gy)

        open_heap_start= [] # A*에서 탐색 후보 노드들을 저장하는 우선순위 
        open_heap_goal = []
        heapq.heappush(open_heap_start, (0 + self.heuristic(start_idx, goal_idx), 0, start_idx)) # START 인덱스 힙에 넣음
        heapq.heappush(open_heap_goal, (0 + self.heuristic(start_idx, goal_idx), 0, goal_idx))
        came_from_start = {}
        came_from_goal = {}
        cost_so_far_start = {start_idx: 0}
        cost_so_far_goal = {goal_idx: 0}

        meet_point = None

        expand_order = cycle(["start", "goal"]) # start와 goal을 번갈아 가며 양방향 a*에서 어느 쪽을 확장할지 결정

        while open_heap_start and open_heap_goal:
            turn = next(expand_order)

            if turn == "start":
                if not open_heap_start:
                    break
                _, cost, current = heapq.heappop(open_heap_start)

                if current in came_from_goal:
                    meet_point = current
                    break

                for neighbor in self._get_neighbors(current):
                    new_cost = cost_so_far_start[current] + self._move_cost(current, neighbor)
                    if neighbor not in cost_so_far_start or new_cost < cost_so_far_start[neighbor]:
                        cost_so_far_start[neighbor] = new_cost
                        priority = new_cost + self.heuristic(neighbor, goal_idx)
                        heapq.heappush(open_heap_start, (priority, new_cost, neighbor))
                        came_from_start[neighbor] = current

            else:  # turn == "goal"
                if not open_heap_goal:
                    break
                _, cost, current = heapq.heappop(open_heap_goal)

                if current in came_from_start:
                    meet_point = current
                    break

                for neighbor in self._get_neighbors(current):
                    new_cost = cost_so_far_goal[current] + self._move_cost(current, neighbor)
                    if neighbor not in cost_so_far_goal or new_cost < cost_so_far_goal[neighbor]:
                        cost_so_far_goal[neighbor] = new_cost
                        priority = new_cost + self.heuristic(neighbor, start_idx)
                        heapq.heappush(open_heap_goal, (priority, new_cost, neighbor))
                        came_from_goal[neighbor] = current


        if meet_point is None:
            return None

        path_from_start = self._reconstruct_path(came_from_start, start_idx, meet_point) # 시작점부터 만나는 점까지의 경로를 담아줌
        path_from_goal = self._reconstruct_path(came_from_goal, goal_idx, meet_point) # 도착점으로부터 만나는 점까지의 경로를 담아줌
        path_from_goal.reverse() # goal부터 시작하는 경로의 순서를 뒤집어줌
        full_path = path_from_start + path_from_goal[1:] # [1:] meetpoint를 제거시켜 중복되지 않게 해줌
        
        interval = min(0.5, max(0.1, self.resolution * 2)) # QPP 방법 사용 interval의 숫자를 키우면 더 촘촘하게 적용
        smooth_path = smooth_path_quintic(full_path, interval=interval)
        return smooth_path # 전체 경로 return

    def _reconstruct_path(self, came_from, start_idx, goal_idx): # 경로 생성
        """Reconstructs path from came_from map."""
        path = []
        current = goal_idx
        while current != start_idx:
            path.append(self._idx_to_pos(current))
            current = came_from.get(current, start_idx)
        path.append(self._idx_to_pos(start_idx))
        path.reverse()
        return path

    def heuristic(self, a, b): # 현재 노드에서 목표 노드까지의 추정 비용 계산
        """Euclidean distance heuristic."""
        return math.hypot((a[0] - b[0]), (a[1] - b[1]))

    def _move_cost(self, a, b):
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        base_cost = self.resolution * math.sqrt(2) if dx and dy else self.resolution

        distance = self._get_distance_from_cache(b)
        penalty = max(0.0, 5.0 - distance)
        return base_cost + penalty
    
    def _get_distance_from_cache(self, b): # 장애물과의 거리를 캐시에 저장하여 다시 계산하지 않도록 해줌(계산속도 단축)
        if b in self.distance_cache:
            # 캐시 항목을 최근 사용으로 갱신
            self.distance_cache.move_to_end(b)
            return self.distance_cache[b]
        else:
            dist = self._min_distance_to_obstacle(b, radius=self.obstacle_search_radius)
            self.distance_cache[b] = dist
            # 캐시 사이즈 초과 시 가장 오래된 항목 제거
            if len(self.distance_cache) > self.distance_cache_max_size:
                self.distance_cache.popitem(last=False)
            return dist
    
    def _min_distance_to_obstacle(self, idx, radius=3): # 가장 근처의 장애물과의 거리를 측정하여 반환
        min_dist = float('inf')
        for dx, dy, dist in self.distance_sorted_deltas:
            if dist > min_dist:
                break  # 더 멀면 탐색 종료
            nx, ny = idx[0] + dx, idx[1] + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if self.grid_map.check_occupancy_from_xy_index(nx, ny):
                    min_dist = dist
        return min_dist if min_dist != float('inf') else radius + 1


    def _get_neighbors(self, idx):
        """Returns traversable neighbors (8-connected).=> 상하좌우, 대각선까지 모두 포함"""
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),  # 격자에서 이웃 셀로의 이동 방향 8개를 튜플로 정의
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dx, dy in directions: # 현재 위치에서 방향 벡터를 더해 이웃 셀 인덱스를 계산
            x2 = idx[0] + dx
            y2 = idx[1] + dy
            if 0 <= x2 < self.width and 0 <= y2 < self.height: # 계산된 인덱스가 격자 맵의 범위 내에 있는지 확인(Out-of-Bounds 방지)
                if not self.grid_map.check_occupancy_from_xy_index(x2, y2): # 해당 셀에 장애물이 없다면 이동 가능하므로 neighbors에 추가
                    neighbors.append((x2, y2))
        return neighbors

    # 셀 단위의 인덱스(격자 기반 A*)를 로봇이 실제로 이동하는 미터 단위의 월드 좌표로 변환
    def _idx_to_pos(self, idx):
        """Converts grid indices back to world coordinates (cell center)."""
        x = self.origin_x + (idx[0] + 0.5) * self.resolution
        y = self.origin_y + (idx[1] + 0.5) * self.resolution
        return [x, y]