import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from geometry_msgs.msg import Point, Quaternion, Pose, PoseStamped
from nav_msgs.msg import Odometry
from grid_map_msgs.msg import GridMap
from rclpy.qos import QoSProfile
from std_msgs.msg import Float32MultiArray
import copy
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from gdm_planning.transformation import (get_transform_between_poses,
        transform_position, calculate_tf, transform_pose, calculate_relative_pose)

from gd_ifc_pkg.msg import GDGGuidancePoint as GuidancePointMsg

class GpSimulator(Node):
    def __init__(self, recevied_gp_len=1):
        super().__init__('gp_simulator')

        # MutuallyExclusiveCallbackGroup 설정
        self.callback_group1 = MutuallyExclusiveCallbackGroup()
        self.callback_group2 = MutuallyExclusiveCallbackGroup()
        self.callback_group3 = MutuallyExclusiveCallbackGroup()
        self.callback_group4 = MutuallyExclusiveCallbackGroup()

        # QoS 설정 (신뢰할 수 있는 전달을 위해 적당한 프로파일 사용)
        #qos_profile = QoSProfile(depth=10)
        qos_profile = 10

        ########### GP simulator
        self.recevied_gp_count = 0
        self.recevied_gp_len = recevied_gp_len
        self.gp = None

        self.odom = None
        self.goal_pose = None

        # 퍼블리셔와 구독자 생성 코드
        self.declare_parameter('position_topic', '/gdg/data/gp')
        self.declare_parameter('odometry_topic', '/zed/zed_node/odom')
        self.declare_parameter('goal_pose_topic', '/goal_pose')
        self.declare_parameter('gridmap_topic', '/gd/mobility_map')
        self.declare_parameter("publish_fake_map", False)  # True 로 설정하면, 빈 GridMap이 10Hz로 출판됨(lidar grid_map 결과와 같은 topic이므로 유의할 것)
        
        # 파라미터 값 가져오기
        self.position_topic = self.get_parameter('position_topic').get_parameter_value().string_value
        self.odometry_topic = self.get_parameter('odometry_topic').get_parameter_value().string_value
        self.goal_pose_topic = self.get_parameter('goal_pose_topic').get_parameter_value().string_value
        self.gridmap_topic = self.get_parameter('gridmap_topic').get_parameter_value().string_value
        self.publish_fake_map = self.get_parameter("publish_fake_map").get_parameter_value().bool_value
        
        # 퍼블리셔 생성 (MutuallyExclusiveCallbackGroup 사용)
        self.position_pub = self.create_publisher(
            GuidancePointMsg, self.position_topic, qos_profile, callback_group=self.callback_group1
        )
        
        # Odometry 구독
        self.odometry_sub = self.create_subscription(
            Odometry, self.odometry_topic, self.odometry_callback, qos_profile, callback_group=self.callback_group2
        )
        
        # 2D Goal Pose from rviz2 구독
        self.goal_pose_sub = self.create_subscription(
            PoseStamped, self.goal_pose_topic, self.goal_pose_callback, qos_profile, callback_group=self.callback_group3
        )
        
        self.get_logger().info(f"GpSimulator Started. Publishing to topic: {self.position_topic}")
        self.get_logger().info(f"Listening to Odometry topic: {self.odometry_topic}")
        self.get_logger().info(f"Listening to Goal Pose topic: {self.goal_pose_topic}")
        self.get_logger().info("Run to echo topic: $ ros2 topic echo /gdg/data/gp |grep -A 23 etype")
        
        ########### Gridmap simulator
        # /gd/mobility_map 토픽에 GridMap 메시지를 출판하는 퍼블리셔 생성
        self.map_pub = self.create_publisher(
            GridMap, self.gridmap_topic, qos_profile, callback_group=self.callback_group4
        )
        
        # 0.1초(10Hz)마다 타이머 호출
        timer_period = 0.1  # 초 (10Hz)
        if self.publish_fake_map:
            self.timer = self.create_timer(timer_period, self.publish_map, callback_group=self.callback_group4)

        # 초기 GridMap 메시지 설정
        self.map_msg = GridMap()
        self.map_msg.info.resolution = 0.2  # 그리드 해상도 설정
        self.map_msg.info.length_x = 200.0  # 맵의 X 길이 설정
        self.map_msg.info.length_y = 200.0  # 맵의 Y 길이 설정

        # 'traversable'이라는 레이어 이름 추가
        self.map_msg.layers.append('traversable')        

        # 0으로 초기화된 데이터를 설정
        array_size = int(self.map_msg.info.length_x * self.map_msg.info.length_y)
        self.data_layer = Float32MultiArray()
        self.data_layer.data = [0.0] * array_size  # 0으로 초기화된 배열

        # 메시지에 레이어 추가
        self.map_msg.data.append(self.data_layer)

    def publish_map(self):
        # 현재 시간을 타임스탬프로 설정
        self.map_msg.header.stamp = self.get_clock().now().to_msg()

        # 메시지 출판
        self.map_pub.publish(self.map_msg)
        #self.get_logger().info('Publishing mobility map at 10Hz')

    def odometry_callback(self, msg): #테스트용
        self.odom = copy.deepcopy(msg)

    def get_zero_odom(self):
        zero_odom = Odometry()
        
        # 메시지 헤더 설정
        zero_odom.header.stamp = self.get_clock().now().to_msg()
        zero_odom.header.frame_id = 'camera_init'  # from line 630 in FAST_LIO/src/laserMapping.cpp
        zero_odom.child_frame_id = 'base_link'
        
        # 위치를 원점으로 설정
        zero_odom.pose.pose.position.x = 0.0
        zero_odom.pose.pose.position.y = 0.0
        zero_odom.pose.pose.position.z = 0.0
        
        # 오리엔테이션을 원점으로 설정 (Quaternion(0,0,0,1)은 회전이 없는 상태를 나타냄)
        zero_odom.pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        
        # 속도를 0으로 설정
        zero_odom.twist.twist.linear.x = 0.0
        zero_odom.twist.twist.linear.y = 0.0
        zero_odom.twist.twist.linear.z = 0.0
        zero_odom.twist.twist.angular.x = 0.0
        zero_odom.twist.twist.angular.y = 0.0
        zero_odom.twist.twist.angular.z = 0.0
        
        # /zero_odom 토픽으로 출판
        return zero_odom

    def get_zero_pose(self):
        pose_msg = PoseStamped()
        pose_msg.pose.position.x = 0.0
        pose_msg.pose.position.y = 0.0
        pose_msg.pose.position.z = 0.0
        pose_msg.pose.orientation.w = 1.0
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = 0.0
        return pose_msg.pose

    def goal_pose_callback(self, msg):
        if self.odom is None:
            return
        self.goal_pose = copy.deepcopy(msg)
        
        relative_pose = Pose()

        if msg.header.frame_id in ["camera_init"] :
            # odom 좌표계(camera_init frame)를 기준으로 goal_pose의 상대 위치 계산
            relative_pose = calculate_relative_pose(src_pose=self.odom.pose.pose, tgt_pose=self.goal_pose.pose)
        else:  # msg.header.frame_id in ["body", "base_link", "robot"]
            relative_pose = self.goal_pose.pose

        # /gdg/data/gp 토픽으로 출판
        # GuidancePointMsg 메시지 생성 및 퍼블리시
        if self.gp is None:
            self.gp = GuidancePointMsg()
            self.gp.odometry = self.odom

        self.gp.position.append(relative_pose.position)
        self.gp.orientation.append(relative_pose.orientation)
        self.recevied_gp_count += 1
        self.gp.gp_type.append(0)
        self.get_logger().info(f"gp_sim_cnt[{self.recevied_gp_count}] [{self.recevied_gp_count}](x,y) = ({relative_pose.position.x:.2f},{relative_pose.position.y:.2f})")

        # self.recevied_gp_len 갯수 만큼 gp가 추가되면 출판하고, gp 초기화
        if self.recevied_gp_count >= self.recevied_gp_len:
            self.position_pub.publish(self.gp)
            self.gp = None
            self.recevied_gp_count = 0
            self.get_logger().info(f"Publish gp from 2D Goal Pose of rviz2")

        # print(f"Pub. Pose: x,y = ({gp.position[0].x}, {gp.position[0].y}),"
        #       f"Orientation = ({gp.orientation[0].x}, {gp.orientation[0].y}, {gp.orientation[0].z}, {gp.orientation[0].w})")


def main(args=None):
    rclpy.init(args=args)
    
    # GpSimulator 노드 생성
    gp_simulator_node = GpSimulator(recevied_gp_len=1)

    # MultiThreadedExecutor 생성
    executor = MultiThreadedExecutor()

    # 노드를 실행할 executor에 추가
    executor.add_node(gp_simulator_node)

    try:
        # executor 실행
        executor.spin()
    finally:
        gp_simulator_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

