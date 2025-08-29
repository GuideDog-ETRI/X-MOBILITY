import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path as PathMsg
from grid_map_msgs.msg import GridMap as GridMapMsg
# from gd_msgs.msg import GuidancePoint as GuidancePointMsg 
from geometry_msgs.msg import PoseStamped as PoseStampedMsg
from gdm_planning.planning.informed_rrt_star import InformedRRTStar 
import numpy as np
from gdm_planning.angle import quaternion_to_euler
from gdm_planning.grid_map import GridMap
from pyquaternion import Quaternion

# std_msgs/msg/Header header # timestamp & id of coordinate system
# uint64 id # id of gp (일련번호: 0, 1, 2, 3, ...)
# geometry_msgs/Point[] position # list of GP position
# geometry_msgs/Quaternion[] orientation # list of desired orientation at GP
# uint8[] gp_type # list of GP type
# uint8[] etype # list of next edge type of GP
# uint8 cur_etype # edge type from current position to GP[0]
# bool is_absolute=true # absolute or relative
class GDMPathTrackingNode(Node):
    def __init__(self):
        super().__init__('gdm_path_tracking_node')
        self.subscriber_map = self.create_subscription(
            PathMsg,
            'gdm/local_plan',
            self.local_plan_callback,
            10)
        self.publisher_cmd_vel = self.create_publisher(PathMsg, 'gdm/cmd_vel', 10)
        
        self.plan_dt = 0.1
        self.max_iter = 100 
        self.expand_dist = 2.0
        self.map_resol = 0.2
        self.map_center_x = 0
        self.map_center_y = 0
        self.map_size_x = 200
        self.map_size_y = 200
        
        self.timer = self.create_timer(self.path_tracking_dt, self.path_tracking_loop)     

    def local_plan_callback(self, msg):
        return
        


    def path_tracking_loop(self): # if new gp or new map
        # Implement Informed RRT algorithm
        # This is a placeholder for the actual Informed RRT implementation
        path = PathMsg()
        self.mobility_map = GridMap(width=self.map_size_x, height=self.map_size_y, resolution=self.map_resol, 
                                 center_x=self.map_center_x, center_y=self.map_center_y, init_val=0.0)
        pol_x = [1, 1, 4, 4, 1]
        pol_y = [1, 4, 4, 1, 1]
        self.mobility_map.set_value_from_polygon(pol_x=pol_x, pol_y=pol_y, val=2)    
        pol_x = [-3, -3, 3, 3]
        pol_y = [5, 8, 8, 5]
        self.mobility_map.set_value_from_polygon(pol_x=pol_x, pol_y=pol_y, val=2)
        pol_x = [3, 3, 5, 5]
        pol_y = [1, 8, 8, 1]
        self.mobility_map.set_value_from_polygon(pol_x=pol_x, pol_y=pol_y, val=2)


        start = [0, 0]
        self.gps = []
        self.gps.append([10, 0, 0])
        self.gps.append([15, 0, 0])
        self.gps.append([20, 0, 0])
        for idx, gp in enumerate(self.gps):
            # Path Planning 
            # Not Viable GP
            # - GP on the Obstacle (Within threshold)
            # - No path to GP
            # - GP outside of map
            print(gp[0], gp[1])
            x_idx, y_idx, valid = self.mobility_map.get_xy_index_from_xy_pos(x_pos=gp[0], y_pos=gp[1])
            if not valid:
                print("GP outside map.")
                continue
            elif self.mobility_map.check_occupancy_from_xy_index(x_idx, y_idx):
                print("GP on the obstacle.")
                continue

            planner = InformedRRTStar(start=start, goal=[gp[0], gp[1]], 
                                      max_iter=self.max_iter, expand_dist=self.expand_dist,
                                      rand_area=[self.map_center_x - self.map_size_x/2.0, self.map_center_x + self.map_size_x/2.0, 
                                      self.map_center_y - self.map_size_y/2.0, self.map_center_y + self.map_size_y/2.0],
                                      gridmap=self.mobility_map)
            path = planner.plan(animation=False)
            if path is None:
                print("No path to GP found.")
                continue
            break
        
        path_msg = PathMsg()
        path_msg.header.frame_id = "robot"
        path_msg.header.stamp = self.get_clock().now().to_msg()
        for pose in reversed(path):
            pose_msg = PoseStampedMsg()
            pose_msg.pose.position.x = pose[0]
            pose_msg.pose.position.y = pose[1]
            pose_msg.pose.position.z = 0
            # Quaternion() Quaternion(axis=[1, 0, 0], angle=3.14159265)
            # pose_msg.pose.orientation.w = 
            # pose_msg.pose.orientation.x
            # pose_msg.pose.orientation.y
            # pose_msg.pose.orientation.z
            path_msg.poses.append(pose_msg)
        self.publisher_plan.publish(path_msg)


    def convert_to_ros_path(self, path):
        ros_path = PathMsg()
        ros_path.header.frame_id = "mobility_map"
        for point in path:
            pose = PoseStampedMsg()
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.position.z = 0.0  # Adjust according to elevation map
            ros_path.poses.append(pose)
        return ros_path

def main(args=None):
    rclpy.init(args=args)
    node = GDMPlanningNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()