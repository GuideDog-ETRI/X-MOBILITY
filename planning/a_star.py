import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))


from gdm_planning.angle import rot_mat_2d
from gdm_planning.grid_map import GridMap

import heapq
import math

class AStarPlanner:
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
        self.width = grid_map.width
        self.height = grid_map.height
        self.resolution = grid_map.resolution
        self.center_x = grid_map.center_x
        self.center_y = grid_map.center_y
        # Compute world coordinates of lower-left grid corner
        self.origin_x = self.center_x - (self.width * self.resolution) / 2.0
        self.origin_y = self.center_y - (self.height * self.resolution) / 2.0

        self.start = start
        self.goal = goal

    def plan(self, animation=False):
        """
        Plans a path from start to goal.
        :param start: [x, y] in world coordinates
        :param goal: [x, y] in world coordinates
        :return: List of [x, y] waypoints in world coordinates or None if no path.
        """
        # Convert to grid indices
        sx, sy, valid_s = self.grid_map.get_xy_index_from_xy_pos(self.start[0], self.start[1])
        gx, gy, valid_g = self.grid_map.get_xy_index_from_xy_pos(self.goal[0], self.goal[1])
        if not valid_s or not valid_g:
            return None
        start_idx = (sx, sy)
        goal_idx = (gx, gy)

        open_heap = []
        heapq.heappush(open_heap, (0 + self.heuristic(start_idx, goal_idx), 0, start_idx))
        came_from = {}
        cost_so_far = {start_idx: 0}

        while open_heap:
            _, cost, current = heapq.heappop(open_heap)

            if current == goal_idx:
                return self._reconstruct_path(came_from, start_idx, goal_idx) # 도착했다면 start부터 goal까지 경로를 return

            for neighbor in self._get_neighbors(current): # 현재 노드 주변의 이웃 노드들을 가져옴
                new_cost = cost_so_far[current] + self._move_cost(current, neighbor) # 현재노드 current까지의 누적비용과 neighbor까지 가는 이동 비용을 더해서 새로운 누적 비용 계산
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]: # 이 이웃노드를 처음 방문하거나 이미 방문한 코드지만 이번 경로가 더 짧은 경우
                    cost_so_far[neighbor] = new_cost # 누적 비용을 저장 또는 갱신
                    # add_cost = dist_from_start + dist_to_goal
                    line_cost = 0 # 이코드부터 아래 line_cost까지는 직선 경로를 우선시 하기 위한 코드
                    # current_position = self._idx_to_pos(neighbor) 
                    dist_from_start = abs(self.heuristic(neighbor,start_idx))
                    dist_from_goal = abs(self.heuristic(neighbor,goal_idx))
                    line_cost = dist_from_start + dist_from_goal
                    # print(f"{dist_from_start},{dist_from_goal}")

                    priority = new_cost + self.heuristic(neighbor, goal_idx) + line_cost
                    heapq.heappush(open_heap, (priority, new_cost, neighbor))
                    came_from[neighbor] = current

        return None

    def _reconstruct_path(self, came_from, start_idx, goal_idx):
        """Reconstructs path from came_from map."""
        path = []
        current = goal_idx
        while current != start_idx:
            path.append(self._idx_to_pos(current))
            current = came_from.get(current, start_idx)
        path.append(self._idx_to_pos(start_idx))
        path.reverse()
        return path

    def heuristic(self, a, b):
        """Euclidean distance heuristic."""
        return math.hypot((a[0] - b[0]), (a[1] - b[1]))

    def _move_cost(self, a, b):
        """Cost between adjacent cells (accounts for diagonal moves)."""
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        if dx and dy:
            return self.resolution * math.sqrt(2)
        return self.resolution

    def _get_neighbors(self, idx):
        """Returns traversable neighbors (8-connected)."""
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dx, dy in directions:
            x2 = idx[0] + dx
            y2 = idx[1] + dy
            if 0 <= x2 < self.width and 0 <= y2 < self.height:
                if not self.grid_map.check_occupancy_from_xy_index(x2, y2):
                    neighbors.append((x2, y2))
        return neighbors

    def _idx_to_pos(self, idx):
        """Converts grid indices back to world coordinates (cell center)."""
        x = self.origin_x + (idx[0] + 0.5) * self.resolution
        y = self.origin_y + (idx[1] + 0.5) * self.resolution
        return [x, y]

class Path:
    """
    RRT Node
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.path_x = []
        self.path_y = []
        self.parent = None
