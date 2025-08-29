import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from geometry_msgs.msg import Point, Pose, PoseStamped
from geometry_msgs.msg import Quaternion as QuaternionMsg
from sensor_msgs.msg import LaserScan as LaserScanMsg
from nav_msgs.msg import Path as PathMsg
from geometry_msgs.msg import Twist as TwistMsg
from grid_map_msgs.msg import GridMap as GridMapMsg
from nav_msgs.msg import Odometry as OdometryMsg
# from gd_msgs.msg import GuidancePoint as GuidancePointMsg 
from geometry_msgs.msg import PoseStamped as PoseStampedMsg
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from gdm_planning.planning.informed_rrt_star import InformedRRTStar
from gdm_planning.planning.a_star import AStarPlanner
from gdm_planning.planning.yunju import YUNJUplanner
# from gdm_planning.tracking.pure_pursuit import (State, TargetCourse, proportional_control, pure_pursuit_steer_control)
#from gdm_planning.tracking.pure_pursuit_biwheel import (DiffState, TargetCourse, PurePursuitController)
#from gdm_planning.tracking.mpi import (State, TargetCourse, mppi_control)
from gdm_planning.tracking.x_mobility_navigator import XMobilityNavigator

# from gdm_planning.tracking.regulated_pure_pursuit import (State, TargetCourse, proportional_control, pure_pursuit_steer_control)
import numpy as np
from gdm_planning.transformation import (get_transform_between_poses, transform_position,
        calculate_tf, transform_pose, calculate_relative_pose, get_zero_pose,
        quaternion_from_euler) #, quaternion_to_euler)
from gdm_planning.angle import quaternion_to_euler
from gdm_planning.grid_map import GridMap
from pyquaternion import Quaternion
import copy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import tf2_ros
from scipy.spatial.transform import Rotation as R
from collections import deque

from geometry_msgs.msg import TransformStamped
from gd_ifc_pkg.msg import GDGGuidancePoint as GuidancePointMsg
from gd_ifc_pkg.msg import GDMStatus as GDMStatusMsg

import threading
import time
# from ipdb import set_trace as bp

import pygame

# from playsound import playsound

import math
from scipy.ndimage import uniform_filter1d

# std_msgs/msg/Header header # timestamp & id of coordinate system
# uint64 id # id of gp (일련번호: 0, 1, 2, 3, ...)
# geometry_msgs/Point[] position # list of GP position
# geometry_msgs/Quaternion[] orientation # list of desired orientation at GP
# uint8[] gp_type # list of GP type
# uint8[] etype # list of next edge type of GP
# uint8 cur_etype # edge type from current position to GP[0]
# bool is_absolute=true # absolute or relative

class GDMFSM():
    def __init__(self):
        self.STATE_WAITING_GP = "waiting_gp"
        self.STATE_GOT_NEW_GP = "listened_gp"
        self.STATE_PLANNED_PATH = "planned_path"
        self.STATE_FOLLOWING_PATH = "following_path"
        self.STATE_ALIGN_ANGLE_START = "align_angle_start" # 수정_ align_angle을 start와 end 두개로 나눔
        self.STATE_ALIGN_ANGLE_END = "align_angle_end"
        self.set_current_state(self.STATE_WAITING_GP)

    def get_current_state(self):
        return self.current_state

    def set_current_state(self, state):
        self.current_state = state

    def is_current_state(self, state):
        if self.current_state == state:
            return True
        else:
            return False

class GDMPlanningNode(Node):

    def __init__(self):
        super().__init__('gdm_path_planning_node')
        # --- 여기에 x-mobility 네비게이터 인스턴스 ---

        # 1) X-Mobility 노드 인스턴스
        #self.navigator = XMobilityNavigator()

        # 2) 파라미터 선언 (한 번만)
        #self.declare_parameter('topic_cmd_vel', '/j100_0778/cmd_vel')

        # 3) 퍼블리셔, 구독자 생성
        #topic_cmd = self.get_parameter('topic_cmd_vel').get_parameter_value().string_value
        qos_reliable = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                                  durability=QoSDurabilityPolicy.VOLATILE,
                                  depth=10)
        #self.publisher_cmd_vel = self.create_publisher(TwistMsg, topic_cmd, qos_reliable)

        qos_xmob = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                              durability=QoSDurabilityPolicy.VOLATILE,
                              depth=1)
        self.xmob_sub = self.create_subscription(
            TwistMsg,
            '/cmd_vel',
            self._on_xmob_cmd,
            qos_xmob
        )

##### 파라미터 선언
        # Topic and general value
        self.declare_parameter('world_frame_id', 'map')
        self.declare_parameter('topic_local_plan', 'gdm/local_plan')
        self.declare_parameter('topic_local_plan_camera_init', 'gdm/local_plan_camera_init')
        self.declare_parameter('topic_local_plan_on_map', 'gdm/local_plan_on_map')
        self.declare_parameter('topic_received_gp_marker', 'gdm/received_gp_marker')
#        self.declare_parameter('topic_cmd_vel', '/j100_0135/cmd_vel')  # gdg jackal
        self.declare_parameter('topic_cmd_vel', '/j100_0778/cmd_vel')  # gdm jackal

        self.declare_parameter('mobility_map_topic', '/gd/mobility_map')
        self.declare_parameter('gdg_gp_topic', '/gdg/data/gp')
        self.declare_parameter('odom_topic', '/zed/zed_node/odom')
        self.declare_parameter("lidar_scan_topic", "/ouster/scan")
        self.declare_parameter("lidar_scan_sim_topic", "/j100_0778/sensors/lidar3d_0/scan")

        self.declare_parameter('max_linear_speed', 0.4)  # 최대 선형 속도 (m/s), 0.4 max speed in joystick's cmd_vel(from experiment)
        self.declare_parameter('max_angular_speed', 0.6) # 최대 각속도 (rad/s),  0.6 max speed in joystick's cmd_vel(from experiment)
        self.declare_parameter('tracking_max_sec', 10.0) # Exit tracking loop after this sec in every tracking loop
        self.declare_parameter('dist2goal_max_sec', 10.0) 

        # Pure_Pursuit Tracking
        self.declare_parameter('pp_k', 1.25)  # look forward gain
        self.declare_parameter('pp_Lfc', 1.5)  # default 0.3[m] look-ahead distance
        self.declare_parameter('pp_Kp', 2.4)  # 1.6  # speed proportional gain
        self.declare_parameter('pp_Ka', 1.6)  # angular velocity propotional gain
        self.declare_parameter('pp_regulating_Ka', 1.0)  # Adaptive Ka. Regulated angular gain. Output near sub-goal: low angle, far sub-goal: large angle. 
        self.declare_parameter('pp_dt', 0.1)  # [s] callback time tick
        self.declare_parameter('pp_WB', 0.000001)  # [m] wheel base of vehicle (jackal 0.26), Distance between front-wheel and rear-whell
        self.declare_parameter('pp_show_animation', False)
        self.declare_parameter('pp_Kdh', 1.0)

        # Path Planning

        self.plan_success = False
        
        self.declare_parameter("plan_dt", 2.0)
        self.declare_parameter("max_iter", 1000)
        self.declare_parameter("epsilon", 0.2)
        self.declare_parameter("expand_dist", 0.2)
        self.declare_parameter("map_resol", 0.2)
        self.declare_parameter("map_center_x", 0)
        self.declare_parameter("map_center_y", 0)
        self.declare_parameter("map_size_x", 200)
        self.declare_parameter("map_size_y", 200)
        
        # Path Tracking
        self.declare_parameter("track_dt", 0.1)
        self.declare_parameter("tgt_speed_default", 1.0)
        self.declare_parameter("waypoint_ddist", 0.1)
        self.declare_parameter("goal_stop_distance", 0.3)
        self.declare_parameter("blocking_time_rcv_gp", 2)
        self.declare_parameter("track_loop_count", 0)
        self.declare_parameter("gp_modify_distance", 5.0)


        ##### 파라미터 가져오기
        self.world_frame_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        # Topic and general value

        topic_received_gp_marker = self.get_parameter('topic_received_gp_marker').get_parameter_value().string_value
        topic_local_plan = self.get_parameter('topic_local_plan').get_parameter_value().string_value
        topic_local_plan_camera_init = self.get_parameter('topic_local_plan_camera_init').get_parameter_value().string_value
        topic_local_plan_on_map = self.get_parameter('topic_local_plan_on_map').get_parameter_value().string_value
        topic_cmd_vel = self.get_parameter('topic_cmd_vel').get_parameter_value().string_value

        self.mobility_map_topic = self.get_parameter('mobility_map_topic').get_parameter_value().string_value
        self.gdg_gp_topic = self.get_parameter('gdg_gp_topic').get_parameter_value().string_value
        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.max_linear_speed = self.get_parameter('max_linear_speed').get_parameter_value().double_value
        self.max_angular_speed = self.get_parameter('max_angular_speed').get_parameter_value().double_value
        self.tracking_max_sec = self.get_parameter('tracking_max_sec').get_parameter_value().double_value
        self.lidar_scan_topic = self.get_parameter("lidar_scan_topic").get_parameter_value().string_value
        self.lidar_scan_sim_topic = self.get_parameter("lidar_scan_sim_topic").get_parameter_value().string_value

        # Pure_Pursuit Tracking
        self.pp_k = self.get_parameter('pp_k').get_parameter_value().double_value
        self.pp_Lfc = self.get_parameter('pp_Lfc').get_parameter_value().double_value
        self.pp_Kp = self.get_parameter('pp_Kp').get_parameter_value().double_value
        self.pp_Ka = self.get_parameter('pp_Ka').get_parameter_value().double_value
        self.pp_regulating_Ka = self.get_parameter('pp_regulating_Ka').get_parameter_value().double_value
        self.pp_dt = self.get_parameter('pp_dt').get_parameter_value().double_value
        self.pp_WB = self.get_parameter('pp_WB').get_parameter_value().double_value
        self.pp_show_animation = self.get_parameter('pp_show_animation').get_parameter_value().bool_value
        self.pp_Kdh = self.get_parameter('pp_Kdh').get_parameter_value().double_value

        # Path Planning
        self.plan_dt = self.get_parameter("plan_dt").get_parameter_value().double_value
        self.max_iter = self.get_parameter("max_iter").get_parameter_value().integer_value
        self.epsilon = self.get_parameter("epsilon").get_parameter_value().double_value
        self.expand_dist = self.get_parameter("expand_dist").get_parameter_value().double_value
        self.map_resol = self.get_parameter("map_resol").get_parameter_value().double_value
        self.map_center_x = self.get_parameter("map_center_x").get_parameter_value().double_value
        self.map_center_y = self.get_parameter("map_center_y").get_parameter_value().double_value
        self.map_size_x = self.get_parameter("map_size_x").get_parameter_value().integer_value
        self.map_size_y = self.get_parameter("map_size_y").get_parameter_value().integer_value
        
        # Path Tracking

        self.track_dt = self.get_parameter("track_dt").get_parameter_value().double_value
        self.target_speed_default = self.get_parameter("tgt_speed_default").get_parameter_value().double_value
        self.target_speed = self.target_speed_default  # [m/s]
        self.waypoint_ddist = self.get_parameter("waypoint_ddist").get_parameter_value().double_value
        self.goal_stop_distance = self.get_parameter("goal_stop_distance").get_parameter_value().double_value
        self.blocking_time_rcv_gp = self.get_parameter("blocking_time_rcv_gp").get_parameter_value().double_value
        self.track_loop_count = self.get_parameter("track_loop_count").get_parameter_value().integer_value
        self.gp_modify_distance = self.get_parameter("gp_modify_distance").get_parameter_value().double_value

        self.gps_start = (0.0, 0.0, 0.0)  # 초기 시작 yaw 설정용 수정


        self.goal_stop_distance_normal = self.goal_stop_distance
        self.goal_stop_distance_strict = 0.2
        #self.state_now = DiffState(x=0.0, y=0.0, yaw=0.0, v=0.0, w=0.0, dt=self.pp_dt)
        #self.controller = PurePursuitController(k=self.pp_k, Lfc=self.pp_Lfc, Kp_speed=self.pp_Kp, Kp_heading=self.pp_Ka, Kd_heading=self.pp_Kdh, dt=self.pp_dt)
        #self.state_now = State(x=0.0, y=0.0, yaw=0.0, v=0.0, w=0.0, dt=self.pp_dt)
        #self.trajectory = TargetCourse(cx=[], cy=[], k=self.pp_k, Lfc=self.pp_Lfc)

        #self.prev_ind = 0
        #self.time = 0.0
        # Tracking loop count 초기화
        self.track_loop_count = 0
        
        self.declare_parameter('drive_cmd_topic', '/j100_0778/cmd_vel')          # 로봇 최종 출력
        self.declare_parameter('xmob_cmd_topic',   '/x_mobility/cmd_vel')        # X-Mobility가 내는 입력

        topic_cmd = self.get_parameter('drive_cmd_topic').get_parameter_value().string_value
        xmob_topic = self.get_parameter('xmob_cmd_topic').get_parameter_value().string_value

        qos_reliable = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                                durability=QoSDurabilityPolicy.VOLATILE, depth=10)

        self.publisher_cmd_vel = self.create_publisher(TwistMsg, topic_cmd, qos_reliable)

        qos_xmob = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                            durability=QoSDurabilityPolicy.VOLATILE, depth=1)
        self.xmob_sub = self.create_subscription(
            TwistMsg, xmob_topic, self._on_xmob_cmd, qos_xmob
        )


        # QoS define
        qos_best_effort = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, depth=3)
        qos_reliable = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, depth=10)

        # 퍼블리셔 생성
        self.publisher_plan = self.create_publisher(PathMsg, topic_local_plan, 10)
        self.publisher_plan2 = self.create_publisher(PathMsg, topic_local_plan_camera_init, 10)
        self.publisher_plan3 = self.create_publisher(PathMsg, topic_local_plan_on_map, 10)
        self.publisher_cmd_vel = self.create_publisher(TwistMsg, topic_cmd_vel, qos_reliable)  # # It must be subscribed

        self.publisher_marker = self.create_publisher(Marker, topic_received_gp_marker, 10)  

        
        

        self.marker = Marker()
        self.marker.header.frame_id = self.world_frame_id
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.fsm = GDMFSM()
        self.queue_size = 10
        self.dist2goal_queue = deque(maxlen=self.queue_size)  # 최대 길이가 10인 deque
        self.dist2goal_end_count = 0 # 수정_

        self.mutex_mobility_map = threading.Lock()
        self.mutex_gp = threading.Lock()
        self.mutex_state = threading.Lock()
        self.mutex_odometry = threading.Lock()
        self.mutex_laserscan = threading.Lock()

        self.group1 = MutuallyExclusiveCallbackGroup()
        self.group2 = MutuallyExclusiveCallbackGroup()
        self.group3 = MutuallyExclusiveCallbackGroup()
        self.group4 = MutuallyExclusiveCallbackGroup()
        self.group5 = MutuallyExclusiveCallbackGroup()
        self.group6 = MutuallyExclusiveCallbackGroup()
        self.group7 = MutuallyExclusiveCallbackGroup()
        self.track_timer = self.create_timer(self.track_dt, self.track_loop, callback_group=self.group1)  
        self.plan_timer = self.create_timer(self.plan_dt, self.plan_loop, callback_group=self.group2)    
        self.subscriber_map = self.create_subscription(
            GridMapMsg,
            self.mobility_map_topic,
            self.mobility_map_callback,
            10, callback_group=self.group3)
        self.subscriber_gp = self.create_subscription(
            GuidancePointMsg,
            self.gdg_gp_topic,
            self.guidance_point_callback,
            10, callback_group=self.group4)
        self.subscriber_odom = self.create_subscription(
            OdometryMsg,
            self.odom_topic,
            self.odometry_callback,
            10, callback_group=self.group5)   
        self.subscriber_scan = self.create_subscription(
            LaserScanMsg,
            self.lidar_scan_topic,
            self.laser_callback,
            10, callback_group=self.group6)   
        
        self.subscriber_scan = self.create_subscription(
            LaserScanMsg,
            self.lidar_scan_sim_topic,
            self.laser_callback2,
            10, callback_group=self.group7)   
        self.minimum_distance = 0.3 # Ouster minimum distance measurement in datasheet
        self.safe_distance = 0.6  # meter
        self.angle_range = 20.0 # +- degrees
        self.emergency_threshold = 2 # number of short rays to make robot stop
        self.is_emergency = False
        self.barked = False
        self.num_bark=3

        self.restart_ready = True
        self.restart_counter = 0

        from ament_index_python.packages import get_package_share_directory
        self.package_share_directory = get_package_share_directory('gdm_planning')
        self.bark_file = self.package_share_directory + '/resource/bark.ogg'
        # self.whistle_file = self.package_share_directory + '/resource/ok_robot.mp3'
        self.whistle_file = self.package_share_directory + '/resource/ok_kart.mp3'
        
        # self.state_now = State(x=0.0, y=0.0, yaw=0.0, v=0.0, w=0.0, dt=self.pp_dt, WB=self.pp_WB)

        # GDM Message publisher
        self.gdm_msg = GDMStatusMsg()
        self.publisher_status = self.create_publisher(GDMStatusMsg, 'gdm/msg/status', 10)
        self.publish_gdm_message( self.fsm.STATE_WAITING_GP, self.is_emergency)


        self.path = None
        self.path_on_world = None
        self.odom_msg = None
        self.map_msg = None
        self.new_map_msg = None
        self.gp_odom_msg = None
        self.gp_type = 0
        self.gps = []

        self.init_yaw = 0.0
        self.final_yaw= 0.0
        self.yaw_gap = 0.0
        self.target_yaw = 0.0

        self.stop_threshold = 0.05

        self.received_gp_error = False

        # 4) 마지막 커맨드 보관용 변수 초기화
        self.xmob_cmd = TwistMsg()

        # … 나머지 파라미터 선언과 콜백 타이머, 퍼블리셔 등 …

    def _on_xmob_cmd(self, msg: TwistMsg):
        # 1) 메시지 저장
        self.xmob_cmd = msg
        # 2) 바로 로봇용 토픽으로 재발행
        self.publisher_cmd_vel.publish(msg)
        

    def publish_gdm_message(self, status, blocked):
        """
        Publish GDM status message.
        :param status: Current status of the GDM.
        :param blocked: Whether the GDM is blocked or not.
        """

        self.gdm_msg.errcode = self.gdm_msg.ERR_NONE  # No error


        self.gdm_msg.status = status
        self.gdm_msg.blocked = blocked
        self.gdm_msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher_status.publish(self.gdm_msg)


    def mobility_map_callback(self, msg):
        if self.odom_msg == None:
            return 

        self.map_msg = copy.deepcopy(msg)
        with self.mutex_odometry:
            self.map_odom_msg = copy.deepcopy(self.odom_msg)
        
        with self.mutex_mobility_map:
            self.mobility_map = GridMap(width=self.map_size_x, height=self.map_size_y, resolution=self.map_resol, 
                                    center_x=self.map_center_x, center_y=self.map_center_y, init_val=0.0, free_val=0.0)
            self.mobility_map.grid_map = self.extract_layer(self.map_msg, 'traversable', self.map_size_x, self.map_size_y)      
            self.mobility_map.flip_y()
            self.mobility_map.flip_x()  


    def guidance_point_callback(self, msg):
        def _compensated_pose(tf_matrix, pos, quat):
            src_pose = Pose()
            src_pose.position = pos
            src_pose.orientation = quat
            tgt_pose = transform_pose(tf_matrix, src_pose)
            return tgt_pose.position, tgt_pose.orientation
        
        try:
            self.gp_type = msg.gp_type[0]
        except:
            pass
        
        if msg.odometry is None:
            return
        # print(self.odom_msg)

        if self.odom_msg is None:
            return
        
        
        self.gp_odom_msg = msg.odometry  # gp 생성 당시의 odom
        with self.mutex_odometry:
            _curr_odom_msg = copy.deepcopy(self.odom_msg)
        _gp_tf_matrix = calculate_tf(src_pose=self.gp_odom_msg.pose.pose, tgt_pose=_curr_odom_msg.pose.pose)

        print(f"gp_type:{self.gp_type}")


        if self.gp_type == 99:      # Stop GP. Do not plan. Do nothing.
            stop_condition = True
            stop_reason = f"GP type indicates stop signal. Do nothing."

            self.fsm.set_current_state(self.fsm.STATE_WAITING_GP)
            self.publish_gdm_message( self.fsm.STATE_WAITING_GP, self.is_emergency)
            return

        with self.mutex_gp:
            self.gps = []  # gp 들
            for _p, _q in zip(msg.position, msg.orientation):
        
                p, q = _compensated_pose(_gp_tf_matrix, _p, _q)
                # self.get_logger().info(f"p: {p}")
                # self.get_logger().info(f"_p: {_p}")
                #self.get_logger().info(f"q: {q}")
                #self.get_logger().info(f"_q: {_q}")
                
                # Stop GP is received. Do not update yaw.
                if (abs(p.x) < self.stop_threshold and abs(p.y) < self.stop_threshold):
                    self.gps.append([p.x, p.y, self.target_yaw])

                else:
                    q_py = Quaternion(w=q.w, x=q.x, y=q.y, z=q.z)
                    rpy = quaternion_to_euler(q_py)
                    self.target_yaw = rpy[2]
                    self.gps.append([p.x, p.y, self.target_yaw])

                    q_current = self.odom_msg.pose.pose.orientation
                    q_current_py = Quaternion(w=q_current.w, x=q_current.x, y=q_current.y, z=q_current.z)
                    rpy_current = quaternion_to_euler(q_current_py)
                    # print(f"***PLANNNED PATH *** Initial Robot Pose: {self.odom_msg.pose.pose.position}, orientation: {math.degrees(rpy[2])}")
                    self.init_yaw = rpy_current[2]

                #if self.fsm.is_current_state(self.fsm.STATE_GOT_NEW_GP):
                self.get_logger().info(f"(compensated) Received GP(x,y,yaw) : {p.x}, {p.y}, {math.degrees(self.target_yaw)}")
                #break  # GPs 너무 많아 1개만 우선 받음 ( 나중에 이줄 삭제 요망)

            if len(self.gps)>=1:
                self.gps = [self.gps[-1]]  # The last one is best gp in teach and replay mode.
                self.get_logger().info(f"Received best GP(x,y,yaw) : {self.gps[0][0]}, {self.gps[0][1]}, {math.degrees(self.target_yaw)}")

            if self.fsm.is_current_state(self.fsm.STATE_FOLLOWING_PATH) or self.fsm.is_current_state(self.fsm.STATE_PLANNED_PATH) or self.fsm.is_current_state(self.fsm.STATE_ALIGN_ANGLE_START) or self.fsm.is_current_state(self.fsm.STATE_ALIGN_ANGLE_END):
                # stop_reason=f"Got new GP while following or planned."
                # self.stop_action(stop_reason)
                self.fsm.set_current_state(self.fsm.STATE_GOT_NEW_GP)
                self.publish_gdm_message( self.fsm.STATE_GOT_NEW_GP, self.is_emergency)
                #return
                #if self.track_loop_count * self.track_dt < self.blocking_time_rcv_gp:
                #    return

            if len(self.gps) > 0:
                print('self.gps > 0')
                with self.mutex_state:
                    self.fsm.set_current_state(self.fsm.STATE_GOT_NEW_GP)
                    self.publish_gdm_message( self.fsm.STATE_GOT_NEW_GP, self.is_emergency)

    def laser_callback(self, msg: LaserScanMsg):
        angle_increment = msg.angle_increment
        # print(f"Angle increment {angle_increment * num_readings}")

        half_range = int(math.radians(self.angle_range) / angle_increment)
        # print(f"{half_range}")

        first_part = msg.ranges[:half_range]  # 배열의 첫 N개
        last_part = msg.ranges[-half_range:]  # 배열의 마지막 N개
    
        # -N에서 N-1까지 순서로 배열 만들기
        target_ranges = first_part + last_part

        # valid_ranges = [r for r in target_ranges if r > self.minimum_distance]

        # minimum_distance = min(valid_ranges)
        # print(f"{minimum_distance}")

        count = sum(self.minimum_distance <= distance <= self.safe_distance for distance in target_ranges)


        # 조건을 만족하는 값이 emergency_threshold 이상인지 판단
        self.is_emergency = count >= self.emergency_threshold

    def laser_callback2(self, msg: LaserScanMsg):
        
        minimum_distance = self.minimum_distance # Ouster minimum distance measurement in datasheet
        safe_distance = self.safe_distance  # meter
        angle_range = self.angle_range

        angle_increment = msg.angle_increment
        # print(f"Angle increment {angle_increment * num_readings}")

        half_range = int(math.radians(angle_range) / angle_increment)
        # print(f"{half_range}")    

        ranges = msg.ranges
        total = len(ranges)

        # 배열 중앙 인덱스
        center = total // 2

        # 중앙을 기준으로 반절씩 앞뒤로 잘라낼 시작/끝 인덱스
        start_idx = max(0, center - half_range)
        end_idx   = min(total, center + half_range)

        # 가운데 2N개 추출
        target_ranges = ranges[start_idx:end_idx]

        count = sum(minimum_distance <= distance <= safe_distance for distance in target_ranges)


        # 조건을 만족하는 값이 emergency_threshold 이상인지 판단
        self.is_emergency = count >= self.emergency_threshold

    def odometry_callback(self, msg):
        with self.mutex_odometry:
            self.odom_msg = copy.deepcopy(msg)

    def extract_layer(self, grid_map_msg, layer_name, size_x, size_y):
        try:
            index = grid_map_msg.layers.index(layer_name)
            layer_data = np.array(grid_map_msg.data[index].data).reshape((size_x, size_y))
            return layer_data
        except ValueError:
            self.get_logger().error(f"Layer {layer_name} not found in the grid map.")
            return None

    def get_zero_PoseStamped(self, frame_id='', stamp=None):
        pose_msg = PoseStampedMsg()
        pose_msg.header.frame_id = frame_id
        if stamp is not None:
            pose_msg.header.stamp = stamp
        else:
            pose_msg.header.stamp = self.get_clock().now().to_msg()

        pose_msg.pose.position.x = 0.0
        pose_msg.pose.position.y = 0.0
        pose_msg.pose.position.z = 0.0
        # Quaternion() Quaternion(axis=[1, 0, 0], angle=3.14159265)
        pose_msg.pose.orientation.w = 1.0
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = 0.0
        return pose_msg
    
    def plan_loop(self):  # if new gp or new map
        self.plan_success = False
        self.get_logger().info(self.fsm.get_current_state())

        if (self.map_msg is None) or (self.odom_msg is None) or (self.gp_odom_msg is None):
            self.plan_success = False
            return

        if not self.fsm.is_current_state(self.fsm.STATE_GOT_NEW_GP):
            self.plan_success = False
            return

        if (abs(self.gps[0][0]) < self.stop_threshold and abs(self.gps[0][1]) < self.stop_threshold):
            stop_reason = f"GP is too close. It seems they send me STOP signal."
            self.stop_action(stop_reason)
            self.plan_success = False
            return

        print(f"==> Starting plan_loop, gp[0]: {self.gps[0][0]}, {self.gps[0][1]}")
        ts_start = time.time()

        with self.mutex_gp:
            with self.mutex_odometry:
                # map 좌표계에서 로봇/GP 계산 (기존 로직 그대로)
                tf_robot_to_map = get_transform_between_poses(
                    src_pose=self.odom_msg.pose.pose, tgt_pose=self.map_odom_msg.pose.pose
                )
                robot_on_map = transform_position(trans=tf_robot_to_map, x=0, y=0, z=0)

                tf_gp_to_map = get_transform_between_poses(
                    src_pose=self.gp_odom_msg.pose.pose, tgt_pose=self.map_odom_msg.pose.pose
                )
                gps = []
                for i in range(len(self.gps)):
                    gps.append(
                        transform_position(trans=tf_gp_to_map,
                                        x=self.gps[i][0], y=self.gps[i][1], z=self.gps[i][2])
                    )

                _path = None
                final_gp_on_map = None

                # ===(이 줄 중요) map -> camera_init 변환 미리 준비===
                tf_map_to_cam = get_transform_between_poses(
                    src_pose=self.map_odom_msg.pose.pose,
                    tgt_pose=self.get_zero_PoseStamped().pose  # camera_init 원점(0,0,0, 단위쿼터니언)
                )

                for idx, gp_on_map in enumerate(gps):
                    x_idx, y_idx, valid = self.mobility_map.get_xy_index_from_xy_pos(
                        x_pos=gp_on_map[0], y_pos=gp_on_map[1]
                    )

                    # RViz 마커는 camera_init으로 찍는다 (map→camera_init 변환만 사용)
                    self.marker.header.stamp = self.get_clock().now().to_msg()
                    self.marker.type = self.marker.SPHERE
                    self.marker.id = 0
                    self.marker.action = self.marker.ADD
                    self.marker.scale.x = 0.25
                    self.marker.scale.y = 0.25
                    self.marker.scale.z = 0.25
                    self.marker.color.r = 0.0
                    self.marker.color.g = 0.0
                    self.marker.color.b = 1.0
                    self.marker.color.a = 0.75
                    _m = transform_position(trans=tf_map_to_cam,
                                            x=float(gp_on_map[0]), y=float(gp_on_map[1]), z=0.0)
                    self.marker.pose.position.x = float(_m[0][0])
                    self.marker.pose.position.y = float(_m[1][0])
                    self.marker.pose.position.z = 1.0
                    self.publisher_marker.publish(self.marker)

                    final_gp_on_map = gp_on_map

                    if not valid:
                        self.plan_success = False
                        if idx == len(gps) - 1:
                            print("GP seems to be blocked.")
                            self.plan_success = False
                        continue
                    elif self.mobility_map.check_occupancy_from_xy_index(x_idx, y_idx):
                        self.plan_success = False
                        print("GP seems to be on the obstacle.")

                        BA_vector = robot_on_map - gp_on_map
                        BA_unit_vector = BA_vector / np.linalg.norm(BA_vector)
                        modify_distance = 0.0
                        while modify_distance <= self.gp_modify_distance:
                            final_gp_on_map = gp_on_map + modify_distance * BA_unit_vector
                            mx_idx, my_idx, mvalid = self.mobility_map.get_xy_index_from_xy_pos(
                                x_pos=final_gp_on_map[0], y_pos=final_gp_on_map[1]
                            )
                            if self.mobility_map.check_occupancy_from_xy_index(mx_idx, my_idx):
                                modify_distance += 0.1
                            else:
                                break

                    self.received_gp_error = False
                    print(f"==> PLAN LOOP: From:{[robot_on_map[0,0], robot_on_map[1,0]]} "
                        f"==> To:{[final_gp_on_map[0,0], final_gp_on_map[1,0]]}")

                    # A*는 map 좌표에서 수행 (기존 그대로)
                    planner = AStarPlanner(
                        start=[robot_on_map[0, 0], robot_on_map[1, 0]],
                        goal=[final_gp_on_map[0, 0], final_gp_on_map[1, 0]],
                        grid_map=self.mobility_map
                    )
                    _path = planner.plan(animation=False)  # map 좌표의 path
                    print("PATH::::::")

                    if _path is None:
                        print("All GPs are not accessable.")
                        self.plan_success = False
                    else:
                        self.plan_success = True
                        print(f"Path found to the goal: {len(_path)} points")
                    break  # 단일 GP 사용

                if self.plan_success is False:
                    print("Path Planning failed. No path found to the goal.")
                    self.fsm.set_current_state(self.fsm.STATE_WAITING_GP)
                    self.publish_gdm_message(self.fsm.STATE_WAITING_GP, self.is_emergency)
                    return

                # =========================
                # 여기부터가 핵심 수정부
                #   1) map 경로 → camera_init으로 한 번만 변환
                #   2) 그걸 그대로 publish + self.path_on_world에 저장
                # =========================
                path_cam = PathMsg()
                path_cam.header.frame_id = "map"
                path_cam.header.stamp = self.odom_msg.header.stamp

                self.path_on_world = []
                if _path is None or len(_path) == 0:
                    ps = self.get_zero_PoseStamped(path_cam.header.frame_id, path_cam.header.stamp)
                    path_cam.poses.append(ps)
                    self.path_on_world = [[0.0, 0.0]]
                else:
                    for (mx, my) in _path:  # _path: map 좌표
                        X = transform_position(trans=tf_map_to_cam, x=float(mx), y=float(my), z=0.0)
                        ps = PoseStampedMsg()
                        ps.header = path_cam.header
                        ps.pose.position.x = float(X[0][0])
                        ps.pose.position.y = float(X[1][0])
                        ps.pose.position.z = 0.0
                        ps.pose.orientation.w = 1.0
                        ps.pose.orientation.x = 0.0
                        ps.pose.orientation.y = 0.0
                        ps.pose.orientation.z = 0.0
                        path_cam.poses.append(ps)
                        self.path_on_world.append([float(X[0][0]), float(X[1][0])])

                # (옵션) camera_init 좌표 위에서 스무딩
                if len(path_cam.poses) >= 5:
                    xs = np.array([p.pose.position.x for p in path_cam.poses])
                    ys = np.array([p.pose.position.y for p in path_cam.poses])
                    xs = uniform_filter1d(xs, size=5, mode='nearest')
                    ys = uniform_filter1d(ys, size=5, mode='nearest')
                    for i, p in enumerate(path_cam.poses):
                        p.pose.position.x = float(xs[i])
                        p.pose.position.y = float(ys[i])
                    self.path_on_world = [[float(xs[i]), float(ys[i])] for i in range(len(xs))]

                # 동일 경로를 두 토픽으로 그대로 발행 (둘 다 camera_init)
                self.publisher_plan.publish(path_cam)
                self.publisher_plan2.publish(path_cam)

                # === 아래의 기존 코드들 삭제/주석 ===
                # - map 좌표를 frame_id="camera_init"로 그대로 퍼블리시하던 블록
                # - tf_camera_init_to_body로 또 변환하던 블록
                # - smooth_path.poses None 체크 등 중복 경로 생성
                # ===============================

                with self.mutex_state:
                    if len(self.path_on_world) <= 1:
                        print(f"Length of path: {len(self.path_on_world)}")
                        print("Path is too short. No path found to the goal.")
                        self.plan_success = False
                        self.fsm.set_current_state(self.fsm.STATE_WAITING_GP)
                        self.publish_gdm_message(self.fsm.STATE_WAITING_GP, self.is_emergency)
                    else:
                        self.fsm.set_current_state(self.fsm.STATE_PLANNED_PATH)
                        self.publish_gdm_message(self.fsm.STATE_PLANNED_PATH, self.is_emergency)
                        self.target_speed = self.target_speed_default
                        self.ctrl_path = []

        self.track_loop_start_ts = time.time()
        ts = time.time() - ts_start
        print(f"plan_loop sec:{ts}")


    def track_loop(self): # if new gp or new map
        # print('track_loop')
        track_path = self.path_on_world
        stop_reason=""

        if track_path == None or len(track_path) == 0:
            return 

        if not (self.fsm.is_current_state(self.fsm.STATE_PLANNED_PATH) or self.fsm.is_current_state(self.fsm.STATE_FOLLOWING_PATH) or self.fsm.is_current_state(self.fsm.STATE_ALIGN_ANGLE_START) or self.fsm.is_current_state(self.fsm.STATE_ALIGN_ANGLE_END)):
            return 

        if self.fsm.is_current_state(self.fsm.STATE_PLANNED_PATH):
            print("Planned Path is received. Start tracking path.")
            with self.mutex_state:
                # Current Linear & Angular Velocity
                print("TRACK_PATH:::::::::::::::")
                print(track_path)
                _ctrl_path = copy.deepcopy(np.array(track_path))
                print("_ctrl_path:::::::::::::::")
                print(_ctrl_path)
                waypoint_ddist = self.waypoint_ddist
                tmp_path = np.empty([0, 2])                
                for idx, (x, y) in enumerate(zip(_ctrl_path[1:, 0], _ctrl_path[1:, 1])):
                    px = _ctrl_path[idx, 0]
                    py = _ctrl_path[idx, 1]
                    if math.sqrt((x-px)*(x-px) + (y-py)*(y-py)) > waypoint_ddist:
                        theta = math.atan2(y-py, x-px) 
                        if abs(theta) < 0.001:
                            theta = 0.001
                        # print(f"{px},{py},{x},{y},{waypoint_ddist},{theta}")

                        dxs = np.arange(px, x, waypoint_ddist * math.cos(theta))
                        dys = np.arange(py, y, waypoint_ddist * math.sin(theta))
                        min_len = min(len(dxs), len(dys))
                        dxs = dxs[:min_len]
                        dys = dys[:min_len]
                        dps = np.stack((dxs.T, dys.T), axis=1)
                        tmp_path = np.vstack((tmp_path, dps))
                    else:
                        dps = np.array([[px, py], [x, y]])
                        tmp_path = np.vstack((tmp_path, dps))
                self.ctrl_path = tmp_path  # _ctrl_path는 전체 path를 waypoint_ddist 이하  단위로 잘라서 list로 만든 결과

                # 경로 생성 후 즉시 FOLLOWING_PATH로 전환하지 않고,
                # 초기 방향 맞춤 상태(STATE_ALIGN_ANGLE_START)로 전환
                #self.fsm.set_current_state(self.fsm.STATE_ALIGN_ANGLE_START)
                #self.publish_gdm_message(self.fsm.STATE_ALIGN_ANGLE_START, self.is_emergency)

                # 현재 로봇 방향(초기 yaw) 저장
                q = self.odom_msg.pose.pose.orientation # 현재 로봇의 자세 받아옴
                q_py = Quaternion(w=q.w, x=q.x, y=q.y, z=q.z)
                rpy = quaternion_to_euler(q_py) # 오일러 각도로 변환
                self.init_yaw = rpy[2] # 현재 로봇의 방향 yaw

                # 목표 방향(yaw) 계산 — 경로의 첫 두 점 방향 각도
                dx = self.ctrl_path[1, 0] - self.ctrl_path[0, 0] # 250721 Crash on here
                dy = self.ctrl_path[1, 1] - self.ctrl_path[0, 1]
                target_yaw = math.atan2(dy, dx)
                self.gps_start = (target_yaw, 0, 0)


        if self.fsm.is_current_state(self.fsm.STATE_FOLLOWING_PATH):
            self.track_loop_count += 1
    
            if len(self.ctrl_path) == 0:
                with self.mutex_state:
                    self.fsm.set_current_state(self.fsm.STATE_WAITING_GP)
                    self.publish_gdm_message( self.fsm.STATE_WAITING_GP, self.is_emergency)
                print('Exit tracking due to invalid path.')
                return
    
            # track_loop_elapsed_ts = time.time() - self.track_loop_start_ts
            # if track_loop_elapsed_ts > self.tracking_max_sec:  # sec 
            #     stop_reason = f"{track_loop_elapsed_ts} (which is over {self.tracking_max_sec}) elapsed. Exit tracking."
            #     self.stop_action(stop_reason)
            #     return

            # last_idx = len(self.ctrl_path) - 1
            # self.trajectory = TargetCourse(cx=self.ctrl_path[:, 0], cy=self.ctrl_path[:, 1], k=self.pp_k, Lfc=self.pp_Lfc)
            path_on_map = []
            tf_gp_to_map = get_transform_between_poses(src_pose=self.gp_odom_msg.pose.pose, tgt_pose=self.map_odom_msg.pose.pose)
            for pt in track_path:
                path_on_map.append(transform_position(trans=tf_gp_to_map, x=pt[0], y=pt[1], z=0)) # for debug
                
            # Path 에 전역좌표계 적용
            tf_robot_to_map = get_transform_between_poses(src_pose=self.odom_msg.pose.pose, tgt_pose=self.map_odom_msg.pose.pose)
            robot_on_map = transform_position(trans=tf_robot_to_map, x=self.odom_msg.pose.pose.position.x, y=self.odom_msg.pose.pose.position.y, z=self.odom_msg.pose.pose.position.z)
            q_robot_orientation = Quaternion(w=self.odom_msg.pose.pose.orientation.w, x=self.odom_msg.pose.pose.orientation.x, y=self.odom_msg.pose.pose.orientation.y, z=self.odom_msg.pose.pose.orientation.z)
            robot_rpy = quaternion_to_euler(q_robot_orientation)
            # self.state_now = State(x=robot_on_map[0], y=robot_on_map[1], yaw=robot_rpy[2] , v=self.odom_msg.twist.twist.linear.x, w=self.odom_msg.twist.twist.angular.z, dt=self.pp_dt, WB=self.pp_WB)
            #self.state_now = State(x=robot_on_map[0], y=robot_on_map[1], yaw=robot_rpy[2] , v=self.odom_msg.twist.twist.linear.x, w=self.odom_msg.twist.twist.angular.z, dt=self.pp_dt)
            #self.state_now = DiffState(x=robot_on_map[0], y=robot_on_map[1], yaw=robot_rpy[2] , v=self.odom_msg.twist.twist.linear.x, w=self.odom_msg.twist.twist.angular.z, dt=self.pp_dt)
            path_msg = PathMsg()
            path_msg.header.frame_id = "camera_init"
            path_msg.header.stamp = self.get_clock().now().to_msg()
            if path_on_map is None:
                self.path_on_map = [[0, 0]]
                pose_msg = self.get_zero_PoseStamped(path_msg.header.frame_id, path_msg.header.stamp)
                path_msg.poses.append(pose_msg)
            else:
                rpath_on_map = [[x[0], x[1]] for x in path_on_map]
                for pose in rpath_on_map:
                    pose_msg = PoseStampedMsg()
                    pose_msg.header.frame_id = path_msg.header.frame_id
                    pose_msg.header.stamp = path_msg.header.stamp
                    pose_msg.pose.position.x = float(pose[0])
                    pose_msg.pose.position.y = float(pose[1])
                    pose_msg.pose.position.z = 0.0
                    # Quaternion() Quaternion(axis=[1, 0, 0], angle=3.14159265)
                    pose_msg.pose.orientation.w = 1.0
                    pose_msg.pose.orientation.x = 0.0
                    pose_msg.pose.orientation.y = 0.0
                    pose_msg.pose.orientation.z = 0.0
                    path_msg.poses.append(pose_msg)
    
            self.publisher_plan3.publish(path_msg)
            #tgt_idx, _ = self.trajectory.search_target_index(self.state_now)
            
            #dist2goal = math.hypot(self.state_now.x - self.trajectory.cx[-1], self.state_now.y - self.trajectory.cy[-1])
            if len(rpath_on_map) == 0:
                return  # 방어 코드

            # 로봇 현재 위치 (map 좌표)
            tf_robot_to_map = get_transform_between_poses(
                src_pose=self.odom_msg.pose.pose, tgt_pose=self.map_odom_msg.pose.pose)
            robot_on_map = transform_position(
                trans=tf_robot_to_map,
                x=self.odom_msg.pose.pose.position.x,
                y=self.odom_msg.pose.pose.position.y,
                z=self.odom_msg.pose.pose.position.z)

            rx = float(robot_on_map[0])
            ry = float(robot_on_map[1])

            # 경로(goal) 마지막 점
            goal_x = float(rpath_on_map[-1][0])
            goal_y = float(rpath_on_map[-1][1])

            dist2goal = math.hypot(rx - goal_x, ry - goal_y)
            
    
            # 큐에 dist2goal 값을 추가
            self.dist2goal_queue.append(dist2goal)
    
            # 큐에 값이 모두 채워지면 평균 변화량을 계산
            dist2goal_trend = self.calculate_average_trend(self.dist2goal_queue)
            if dist2goal_trend >= 0:  # Abnormal
                self.dist2goal_end_count += 1
            else:  # Normal
                self.dist2goal_end_count -= 3
                self.dist2goal_end_count = max(self.dist2goal_end_count, 0)
            
            # if self.track_loop_count % 20 == 0:  # 2 sec. (10hz)
            #     print(f"Tracking<delta:{dist2goal_trend}>[{self.track_loop_count:03d}]: dist2goal:i[{dist2goal:.2f}] meters, {self.state_now.x[0]:.2f}, {self.state_now.y[0]:.2f} ==> {self.trajectory.cx[-1]:.2f}, {self.trajectory.cy[-1]:.2f}")
            #     #print(f"self.path : {self.path}")
            #     #print(f"self.path_on_world : {self.path_on_world}")

            #     q = self.odom_msg.pose.pose.orientation
            #     q_py = Quaternion(w=q.w, x=q.x, y=q.y, z=q.z)
            #     rpy = quaternion_to_euler(q_py)
            #     print(f" robot pose: {self.odom_msg.pose.pose.position}, orientation: {math.degrees(rpy[2])}")

            stop_condition = False

            if self.gp_type == 2:
                self.goal_stop_distance = self.goal_stop_distance_strict

            else:
                self.goal_stop_distance = self.goal_stop_distance_normal

            if dist2goal < self.goal_stop_distance:
                stop_condition = True
                stop_reason = f"Near, dist2goal:{dist2goal}"
            # if last_idx < tgt_idx:
            #     stop_condition = True
            #     stop_reason = f"Last Idx:{tgt_idx}"
    
            if stop_condition:
                self.stop_action(stop_reason)

                # Publish NULL path to request new GP
                if self.gp_type != 2:       # Goal GP일때는 GP 요청 금지
                    null_path_msg = PathMsg()
                    null_path_msg.header.frame_id = "camera_init"
                    null_path_msg.header.stamp = self.get_clock().now().to_msg()
                    pose_msg = self.get_zero_PoseStamped(null_path_msg.header.frame_id, null_path_msg.header.stamp)
                    null_path_msg.poses.append(pose_msg)
                    self.publisher_plan.publish(null_path_msg)

                if self.gp_type == 2 and self.received_gp_error:
                    print(f"Critical!!! Received Goal GP is out of range!!!")
                    # print(f"X:{self.gps[]}")
                    null_path_msg = PathMsg()
                    null_path_msg.header.frame_id = "camera_init"
                    null_path_msg.header.stamp = self.get_clock().now().to_msg()
                    pose_msg = self.get_zero_PoseStamped(null_path_msg.header.frame_id, null_path_msg.header.stamp)
                    null_path_msg.poses.append(pose_msg)
                    self.publisher_plan.publish(null_path_msg)


                return
            
            #v_cmd, w_cmd= mppi_control(self.state_now, self.trajectory, tgt_idx, self.target_speed, N=120, horizon=12, dt=0.1)



            # ai = proportional_control(target=self.target_speed, current=self.state_now.v, Kp=self.pp_Kp)
            # di, _ = pure_pursuit_steer_control(state=self.state_now, trajectory=self.trajectory, pind=tgt_idx,
            #             Kp=self.pp_Kp, Ka=self.pp_Ka, dt=self.pp_dt, WB=self.pp_WB)  #  , regulating_Ka=self.pp_regulating_Ka)
            # self.state_now.update(ai, di)  # Control vehicle

            # print(f"Current| x={self.state_now.x}, y={self.state_now.y}, yaw={math.degrees(self.state_now.yaw)}")
            # print(f"Previous | x={self.trajectory.cx[self.state_now]}, y={self.trajectory.cy[self.prev_ind]}")
            

            #x1 = self.state_now.x
            #y1 = self.state_now.y
            #x2 = self.trajectory.cx[tgt_idx]
            #y2 = self.trajectory.cy[tgt_idx]
            #dx = x2 - x1
            #dy = y2 - y1
            #target_yaw = math.atan2(dy, dx)

            # print(f"Target | x={self.trajectory.cx[tgt_idx]}, y={self.trajectory.cy[tgt_idx]}, yaw={math.degrees(target_yaw)}")

            #yaw_gap = target_yaw - self.state_now.yaw

            #yaw_gap_abs=abs(yaw_gap)

            #if yaw_gap_abs >= math.pi/3:
            #    yaw_gap_abs = math.pi/3

            # print(f"Yaw gap abs: {math.degrees(yaw_gap_abs)}")

            #linear_target_speeed = self.target_speed_default * math.cos(yaw_gap_abs)

            # if linear_target_speeed < 0.0:
            #     linear_target_speeed = 0.0


            # v_cmd, w_cmd, self.prev_ind = self.controller.control(self.state_now, self.trajectory, self.prev_ind, self.target_speed)
            #v_cmd, w_cmd = self.controller.control(self.state_now, self.trajectory, self.prev_ind, linear_target_speeed)
            #self.state_now.update(v_cmd, w_cmd)
    
            #cmd_vel_msg = TwistMsg()
            #X-MOBILITY
            # 1) x-mobility 가 발행한 v, w 꺼내기
            v_cmd = float(self.xmob_cmd.linear.x)
            w_cmd = float(self.xmob_cmd.angular.z)

            # 2) 속도 제한 (기존 제한 로직 재사용)
            v_cmd = self.limit_speed(v_cmd, self.max_linear_speed)
            w_cmd = self.limit_speed(w_cmd, self.max_angular_speed)

            # 3) 기존 퍼블리셔로 그대로 퍼블리시
            cmd_vel_msg = TwistMsg()
            cmd_vel_msg.linear.x  = v_cmd
            cmd_vel_msg.angular.z = w_cmd
            self.publisher_cmd_vel.publish(cmd_vel_msg)

            if self.is_emergency is True: # 막혔음. 이머전시
                cmd_vel_msg.linear.x = 0.0
                cmd_vel_msg.angular.z = 0.0
                self.get_logger().info('Obstacle detected! Stopping the robot.')
                self.restart_ready = False
                self.restart_counter = 0
                
                self.publish_gdm_message( self.fsm.get_current_state(), self.is_emergency)

                if self.barked is False:
                    for i in range (0, self.num_bark):
                        try:
                            play_audio(self.bark_file)
                        except:
                            pass
                        i+=1
                    
                    self.barked = True

            else:  # 이머전시 해제
                if self.restart_ready is False and self.barked is True: # 짖었음. 출발레디는 안됨

                    if self.restart_counter == 0:                       # 출발 레디 시그널 발행
                        # play_audio(self.whistle_file)                 # 카운트다운 사운드
                        self.publish_gdm_message( self.fsm.get_current_state(), self.is_emergency)

                    self.restart_counter +=1
                    print(f"restart_counter: {self.restart_counter}")

                    if self.restart_counter >= 1:   # audio file length + counter*100ms
                        self.restart_ready = True
                        self.barked = False
                        self.restart_counter = 0


                # if self.restart_ready is True:                          # 출발 준비됨
                #     cmd_vel_msg.linear.x = self.limit_speed(float(self.state_now.v), self.max_linear_speed)
                #     if cmd_vel_msg.linear.x < 0.0:
                #         cmd_vel_msg.linear.x = 0.0
                #     cmd_vel_msg.angular.z = self.limit_speed(float(self.state_now.w), self.max_angular_speed)
                if self.restart_ready is True:
                    v_cmd = self.limit_speed(float(self.xmob_cmd.linear.x),  self.max_linear_speed)
                    w_cmd = self.limit_speed(float(self.xmob_cmd.angular.z), self.max_angular_speed)
                    if v_cmd < 0.0:
                        v_cmd = 0.0
                    cmd_vel_msg.linear.x  = v_cmd
                    cmd_vel_msg.angular.z = w_cmd

                    self.barked = False
                    self.restart_ready = True
                    self.restart_counter = 0

            self.publisher_cmd_vel.publish(cmd_vel_msg)

        if self.fsm.is_current_state(self.fsm.STATE_ALIGN_ANGLE_START): # 수정
            self.target_yaw = self.gps_start[0]  # 시작 목표 yaw

            # 현재 yaw 읽기
            q = self.odom_msg.pose.pose.orientation
            q_py = Quaternion(w=q.w, x=q.x, y=q.y, z=q.z)
            rpy = quaternion_to_euler(q_py)
            self.current_yaw = rpy[2]

            self.yaw_gap = self.current_yaw - self.init_yaw
            if self.yaw_gap > math.pi:
                self.yaw_gap -= 2 * math.pi
            elif self.yaw_gap < -math.pi:
                self.yaw_gap += 2 * math.pi

            # print(f"*** STOPPED TRACKING *** Final Robot Pose: {self.odom_msg.pose.pose.position}, orientation: {math.degrees(rpy[2])}")
            # print(f"You've rotated during tracking: {math.degrees(self.yaw_gap)}")

            self.remain_yaw = self.target_yaw - self.current_yaw
            if self.remain_yaw > math.pi:
                self.remain_yaw -= 2 * math.pi
            elif self.remain_yaw < -math.pi:
                self.remain_yaw += 2 * math.pi
            # print(f"Remain yaw: {math.degrees(self.remain_yaw)}")
            # print(f"**************************************************************")


            with self.mutex_gp:
                if (self.remain_yaw < 0.05) and (self.remain_yaw > -0.05):
                    # 정지 명령
                    self.tgt_speed = 0
                    cmd_vel_msg = TwistMsg()
                    cmd_vel_msg.linear.x = 0.0
                    cmd_vel_msg.angular.z = 0.0
                    self.publisher_cmd_vel.publish(cmd_vel_msg)
                    self.tgt_speed = self.target_speed_default  # 속도 초기화


		    # # Prepare tracking
                    self.fsm.set_current_state(self.fsm.STATE_FOLLOWING_PATH)
                    self.publish_gdm_message( self.fsm.STATE_FOLLOWING_PATH, self.is_emergency)
                    self.track_origin_odom_msg = copy.deepcopy(self.odom_msg)  # Save the initial pose of track of every planned path.
                    self.track_loop_count = 0
                    self.dist2goal_end_count = 0
                    self.dist2goal_queue = deque(maxlen=self.queue_size)

                    self.ctrl_path = copy.deepcopy(np.array(track_path))
                    self.target_speed = self.target_speed_default  # Normal로 복구
                    # print(track_path)
                    #self.trajectory = TargetCourse(cx=self.ctrl_path[:, 0], cy=self.ctrl_path[:, 1], k=self.pp_k, Lfc=self.pp_Lfc)
                    self.prev_ind = 0
                    self.time = 0.0

                    self.fsm.set_current_state(self.fsm.STATE_FOLLOWING_PATH)
                    self.publish_gdm_message( self.fsm.STATE_FOLLOWING_PATH, self.is_emergency)
                
                else:
                    cmd_vel_msg = TwistMsg()
                    cmd_vel_msg.linear.x = float(0.0)
                    # angular_velocity = 0.9*self.remain_yaw
                    cmd_vel_msg.angular.z = float(0.25*self.remain_yaw)

                    if cmd_vel_msg.angular.z > 0.0:
                        cmd_vel_msg.angular.z = max(0.5,cmd_vel_msg.angular.z)

                    else:
                        cmd_vel_msg.angular.z = max(0.5,abs(cmd_vel_msg.angular.z)) * -1.0

                    self.publisher_cmd_vel.publish(cmd_vel_msg)

        if self.fsm.is_current_state(self.fsm.STATE_ALIGN_ANGLE_END):
            q = self.odom_msg.pose.pose.orientation
            q_py = Quaternion(w=q.w, x=q.x, y=q.y, z=q.z)
            rpy = quaternion_to_euler(q_py)

            self.final_yaw = rpy[2]

            self.yaw_gap = self.final_yaw - self.init_yaw
            if self.yaw_gap > math.pi:
                self.yaw_gap -= 2 * math.pi
            elif self.yaw_gap < -math.pi:
                self.yaw_gap += 2 * math.pi

            # print(f"*** STOPPED TRACKING *** Final Robot Pose: {self.odom_msg.pose.pose.position}, orientation: {math.degrees(rpy[2])}")
            # print(f"You've rotated during tracking: {math.degrees(self.yaw_gap)}")

            self.remain_yaw = self.gps[0][2] - self.yaw_gap
            if self.remain_yaw > math.pi:
                self.remain_yaw -= 2 * math.pi
            elif self.remain_yaw < -math.pi:
                self.remain_yaw += 2 * math.pi
            # print(f"Remain yaw: {math.degrees(self.remain_yaw)}")
            # print(f"**************************************************************")


            with self.mutex_gp:

                if (self.remain_yaw < 0.05) and (self.remain_yaw > -0.05):
                    self.target_speed = 0
                    cmd_vel_msg = TwistMsg()
                    cmd_vel_msg.linear.x = float(0.0)      
                    cmd_vel_msg.angular.z = float(0.0)
                    self.publisher_cmd_vel.publish(cmd_vel_msg)
                    self.target_speed = self.target_speed_default  # Normal로 복구

                    self.fsm.set_current_state(self.fsm.STATE_WAITING_GP)
                    self.publish_gdm_message( self.fsm.STATE_WAITING_GP, self.is_emergency)
                    print(f"*** STOPPED TRACKING *** Final Robot Pose: {self.odom_msg.pose.pose.position}, orientation: {math.degrees(rpy[2])}")
                    print(f"**************************************************************")
                    return
                
                else:
                    cmd_vel_msg = TwistMsg()
                    cmd_vel_msg.linear.x = float(0.0)
                    # angular_velocity = 0.9*self.remain_yaw
                    cmd_vel_msg.angular.z = float(0.5*self.remain_yaw)

                    if cmd_vel_msg.angular.z > 0.0:
                        cmd_vel_msg.angular.z = max(0.5,cmd_vel_msg.angular.z)

                    else:
                        cmd_vel_msg.angular.z = max(0.5,abs(cmd_vel_msg.angular.z)) * -1.0

                    self.publisher_cmd_vel.publish(cmd_vel_msg)


    def limit_speed(self, value, max_speed):
        if abs(value) > max_speed:
            return max_speed if value > 0 else -max_speed
        return value    

    def stop_action(self, stop_reason="Receviced Stop Command"):
        self.target_speed = 0
        cmd_vel_msg = TwistMsg()
        cmd_vel_msg.linear.x = float(0.0)      
        cmd_vel_msg.angular.z = float(0.0)
        self.publisher_cmd_vel.publish(cmd_vel_msg)
        with self.mutex_state:
            self.get_logger().info(f'Exit tracking. {stop_reason}')
            
            self.get_logger().info(f"Finalize Moving. GP(x,y,yaw) : {self.gps[0][0]}, {self.gps[0][1]}, {math.degrees(self.gps[0][2])}")

            # Temporary ignore final yaw for VTR

            self.fsm.set_current_state(self.fsm.STATE_ALIGN_ANGLE_END)
            self.publish_gdm_message( self.fsm.STATE_ALIGN_ANGLE_END, self.is_emergency)
            # self.fsm.set_current_state(self.fsm.STATE_WAITING_GP)
            # self.publish_gdm_message( self.fsm.STATE_WAITING_GP, self.is_emergency)

    def calculate_average_trend(self, queue):
        """큐의 값들에 대해 평균의 증가/감소 여부를 계산하는 함수"""
        if len(queue) < self.queue_size:
            # 큐에 10개의 값이 채워지지 않은 경우 변화량을 알 수 없음
            return 0

        # 마지막 N/2개의 평균과 그 이전 5개의 평균을 계산
        half_size = int(self.queue_size*0.5)
        first_half_avg = np.mean(list(queue)[:half_size])
        second_half_avg = np.mean(list(queue)[half_size:])

        # 평균 비교: 증가하면 True, 감소하면 False, 같으면 None
        if second_half_avg > first_half_avg:
            return 1
        elif second_half_avg < first_half_avg:
            return -1
        else:
            return 0
        
def play_audio(filepath):
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(filepath)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue
    finally:
        pygame.mixer.quit()  # Release device


# def main(args=None):
#     rclpy.init(args=args)
#     node = GDMPlanningNode()
#     executor = MultiThreadedExecutor() 
#     executor.add_node(node)
#     try:
#         executor.spin()
#     finally:
#         node.destroy_node()
#         executor.shutdown()
#     rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)

    # 1) 기존 GDMPlanningNode
    gdm_node  = GDMPlanningNode()
    # 2) x-mobility 네비게이터 노드
    xmob_node = XMobilityNavigator()

    # 둘 다 동시에 실행할 MultiThreadedExecutor
    executor = MultiThreadedExecutor()
    executor.add_node(gdm_node)
    executor.add_node(xmob_node)

    try:
        executor.spin()
    finally:
        # 종료할 때는 생성 순서의 역순으로 destroy
        xmob_node.destroy_node()
        gdm_node.destroy_node()
        executor.shutdown()
        rclpy.shutdown()



if __name__ == '__main__':
    main()
