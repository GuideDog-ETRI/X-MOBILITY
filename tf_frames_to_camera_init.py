import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Pose, PoseStamped
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Quaternion as QuaternionMsg
from nav_msgs.msg import Path as PathMsg
from geometry_msgs.msg import Twist as TwistMsg
from grid_map_msgs.msg import GridMap as GridMapMsg
from nav_msgs.msg import Odometry as OdometryMsg
# from gd_msgs.msg import GuidancePoint as GuidancePointMsg 
from geometry_msgs.msg import PoseStamped as PoseStampedMsg
from geometry_msgs.msg import PoseStamped
from gdm_planning.planning.informed_rrt_star import InformedRRTStar 
from gdm_planning.tracking.pure_pursuit import * 
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
from tf2_ros import TransformBroadcaster
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from scipy.spatial.transform import Rotation as R
from collections import deque

from geometry_msgs.msg import TransformStamped
from gd_ifc_pkg.msg import GDGGuidancePoint as GuidancePointMsg

import threading
import time

class TFNode(Node):
    def __init__(self):
        super().__init__('TFNode')

        ### Parameters
        # Frame id
        self.declare_parameter("frame_id_odometry", "camera_init")
        self.declare_parameter("child_frame_id_from_odom_to_body", "rbq_body")
        self.declare_parameter("frame_id_point_cloud_clone", "body")
        self.declare_parameter("frame_id_general_map", "map")
        self.declare_parameter("frame_id_general_odom", "odom")
        self.declare_parameter("frame_id_general_base_link", "base_link")
        self.declare_parameter("frame_id_mobility_map_on_camera_init", "mobility_map_on_camera_init")
        self.declare_parameter("frame_id_os_sensor_on_camera_init", "os_sensor_on_camera_init")
        
        # Topic
        self.declare_parameter("topic_odom", "/gdq/msg/gdq_odom")
        self.declare_parameter("topic_point_cloud", "/ouster/points")
        self.declare_parameter("topic_mobility_map", "/gd/mobility_map")
        self.declare_parameter("topic_points_on_camera_init", "/ouster/points_on_camera_init")
        self.declare_parameter("topic_mobility_map_on_camera_init", "/gd/mobility_map_on_camera_init")
        self.declare_parameter("topic_rbq_odom_on_camera_init", "/gdq_odometry")
        
        ### Get Parameters
        # Frame id
        self.frame_id_odometry = self.get_parameter("frame_id_odometry").value
        self.child_frame_id_from_odom_to_body = self.get_parameter("child_frame_id_from_odom_to_body").value
        self.frame_id_point_cloud_clone = self.get_parameter("frame_id_point_cloud_clone").value
        self.frame_id_general_map = self.get_parameter("frame_id_general_map").value
        self.frame_id_general_odom = self.get_parameter("frame_id_general_odom").value
        self.frame_id_general_base_link = self.get_parameter("frame_id_general_base_link").value
        self.frame_id_mobility_map_on_camera_init = self.get_parameter("frame_id_mobility_map_on_camera_init").value
        self.frame_id_os_sensor_on_camera_init = self.get_parameter("frame_id_os_sensor_on_camera_init").value
        
        # Topic
        self.topic_odom = self.get_parameter("topic_odom").value
        self.topic_point_cloud = self.get_parameter("topic_point_cloud").value
        self.topic_mobility_map = self.get_parameter("topic_mobility_map").value
        self.topic_points_on_camera_init = self.get_parameter("topic_points_on_camera_init").value
        self.topic_mobility_map_on_camera_init = self.get_parameter("topic_mobility_map_on_camera_init").value
        self.topic_rbq_odom_on_camera_init = self.get_parameter("topic_rbq_odom_on_camera_init").value

        # Transform 메시지 생성
        self.tf = TransformStamped()

        # StaticTransformBroadcaster 생성
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)

        self.group1 = MutuallyExclusiveCallbackGroup()
        self.group2 = MutuallyExclusiveCallbackGroup()
        self.group3 = MutuallyExclusiveCallbackGroup()
        self.group4 = MutuallyExclusiveCallbackGroup()

        # 정적 변환 발행 함수 호출
        self.publish_static_transform(frame_id=self.frame_id_odometry, child_frame_id=self.frame_id_general_map, tr=[0.0, 0.0, 0.0], rot=[0.0,0.0,0.0,1.0])
        self.publish_static_transform(frame_id=self.frame_id_general_odom, child_frame_id=self.frame_id_general_base_link, tr=[0.0, 0.0, 0.0], rot=[0.0,0.0,0.0,1.0])
        
        # 구독자 정의
        self.subscriber_odom = self.create_subscription(
            OdometryMsg,
            self.topic_odom,
            self.odometry_callback,
            10, callback_group=self.group1)
        
        self.subscriber_points = self.create_subscription(
            PointCloud2,
            self.topic_point_cloud,
            self.point_cloud_callback,
            10, callback_group=self.group2)
        
        self.subscriber_mobility_map = self.create_subscription(
            GridMapMsg,
            self.topic_mobility_map,
            self.mobility_map_callback,
            10, callback_group=self.group3)
        
        # 출판자 정의 (변수 적용)
        self.publisher_points_on_camera_init = self.create_publisher(
            PointCloud2,
            self.topic_points_on_camera_init,
            10, callback_group=self.group2)
        
        self.publisher_mobility_map_on_camera_init = self.create_publisher(
            GridMapMsg,
            self.topic_mobility_map_on_camera_init,
            10, callback_group=self.group3)
        
        self.publisher_rbq_odom_on_camera_init = self.create_publisher(
            OdometryMsg,
            self.topic_rbq_odom_on_camera_init,
            10, callback_group=self.group4)

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # odom.pose.pose 데이터를 저장할 변수
        self.last_odom = None


    def point_cloud_callback(self, msg):
        # frame_id만 변경한 새로운 msg 생성
        clone_msg = copy.deepcopy(msg)
        clone_msg.header.frame_id = self.frame_id_point_cloud_clone
        self.publisher_points_on_camera_init.publish(clone_msg)

    def mobility_map_callback(self, msg):
        # frame_id만 변경한 새로운 msg 생성
        clone_msg = copy.deepcopy(msg)
        clone_msg.header.frame_id = self.frame_id_mobility_map_on_camera_init
        #clone_msg.header.frame_id = 'body'
        self.publisher_mobility_map_on_camera_init.publish(clone_msg)

    def odometry_callback(self, msg):
        # Odometry 데이터 저장
        self.last_odom = copy.deepcopy(msg)
        # 새로운 frame 간의 TF 발행: 새로운 frame의 원점을 odom의 pose로 셋팅
        self.publish_tf(frame_id=self.frame_id_odometry, child_frame_id=self.frame_id_os_sensor_on_camera_init)
        self.publish_tf(frame_id=self.frame_id_odometry, child_frame_id=self.frame_id_mobility_map_on_camera_init)
        self.publish_tf(frame_id=self.frame_id_general_map, child_frame_id=self.frame_id_general_odom)

        # rbq odometry header 교체
        clone_msg = copy.deepcopy(msg)
        clone_msg.header.frame_id = self.frame_id_odometry
        clone_msg.child_frame_id = self.child_frame_id_from_odom_to_body  # 'body or 'rbq_body'
        #clone_msg.header.frame_id = 'body'
        self.publisher_rbq_odom_on_camera_init.publish(clone_msg)

    def publish_tf(self, frame_id='camera_init', child_frame_id='child_frame'):
        if self.last_odom is None:
            return

        # Header 정보
        self.tf.header.stamp = self.get_clock().now().to_msg()
        self.tf.header.frame_id = frame_id 
        self.tf.child_frame_id = child_frame_id

        # odom.pose.pose 데이터를 TF 변환에 반영
        self.tf.transform.translation.x = self.last_odom.pose.pose.position.x
        self.tf.transform.translation.y = self.last_odom.pose.pose.position.y
        self.tf.transform.translation.z = self.last_odom.pose.pose.position.z

        self.tf.transform.rotation.x = self.last_odom.pose.pose.orientation.x
        self.tf.transform.rotation.y = self.last_odom.pose.pose.orientation.y
        self.tf.transform.rotation.z = self.last_odom.pose.pose.orientation.z
        self.tf.transform.rotation.w = self.last_odom.pose.pose.orientation.w

        # TF 변환 발행
        self.tf_broadcaster.sendTransform(self.tf)

    def publish_static_transform(self, frame_id='camera_init', child_frame_id='os_sensor', tr=[0.0, 0.0, 0.0], rot=[0.0,0.0,0.0,1.0]):
        # TransformStamped 메시지 생성
        static_transform_stamped = TransformStamped()

        # Header 설정
        static_transform_stamped.header.stamp = self.get_clock().now().to_msg()
        static_transform_stamped.header.frame_id = frame_id   # 부모 프레임
        static_transform_stamped.child_frame_id = child_frame_id      # 자식 프레임

        # 변환 설정 (translation)
        static_transform_stamped.transform.translation.x = tr[0]  # 예시 값, 필요에 따라 수정
        static_transform_stamped.transform.translation.y = tr[1]
        static_transform_stamped.transform.translation.z = tr[2]

        # 회전 설정 (rotation: quaternion 값)
        static_transform_stamped.transform.rotation.x = rot[0]
        static_transform_stamped.transform.rotation.y = rot[1]
        static_transform_stamped.transform.rotation.z = rot[2]
        static_transform_stamped.transform.rotation.w = rot[3]

        # 정적 변환 발행
        self.tf_static_broadcaster.sendTransform(static_transform_stamped)
        self.get_logger().info(f'Published static transform from {frame_id} to {child_frame_id}')

def main(args=None):
    rclpy.init(args=args)
    node = TFNode()
    executor = MultiThreadedExecutor() 
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        executor.shutdown()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
