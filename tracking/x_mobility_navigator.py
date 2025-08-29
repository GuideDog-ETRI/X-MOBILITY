# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
#import pycuda.autoinit  # pylint: disable=unused-import
import pycuda.driver as cuda
import rclpy
import tensorrt as trt
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs import do_transform_pose
import cv2

IMAGE_TOPIC_NAME = '/j100_0778/sensors/camera_0/color/image'
ODOM_TOPIC_NAME = '/j100_0778/platform/odom/filtered'
CMD_TOPIC_NAME   = '/cmd_vel'
ROUTE_TOPIC_NAME = 'gdm/local_plan_camera_init'
GOAL_TOPIC_NAME = '/goal_pose'
PATH_TOPIC_NAME = '/x_mobility_path'
RUNTIME_PATH = 'runtime_path'
MAPLESS_FLAG = 'is_mapless'

NUM_ROUTE_POINTS = 20 # 경로에서 사용할 최대 지점(point) 수를 20개로 제한 -> 일정 개수로 맞추면 신경망 입력 크기 고정 가능
# Route vector with 4 values representing start and end positions
ROUTE_VECTOR_SIZE = 4 # 한 개의 경로 벡터가 4개의 숫자로 표현 -> 한 개의 경로 벡터가 4개의 숫자로 표현
ROBOT_FRAME = 'camera_init'




# Upsample the points between start and goal. start와 goal 사이 거리가 너무 멀 때, 최대 간격을 넘지 않도록 중간 점들을 만들어서 점들을 촘촘히 연결
def upsample_points(start, goal, max_segment_length):
    x1, y1 = start
    x2, y2 = goal

    # Calculate the Euclidean distance between the two points
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # Handle the case where the start and goal are too close (or identical)
    if distance <= max_segment_length:
        return [start, goal]

    # Determine the number of segments based on the maximum segment length
    num_segments = max(1, int(np.ceil(distance / max_segment_length)))

    # Generate the interpolated points
    interpolated_points = [(x1 + (i / num_segments) * (x2 - x1),
                            y1 + (i / num_segments) * (y2 - y1))
                           for i in range(num_segments + 1)]

    return interpolated_points


class XMobilityNavigator(Node):
    '''X-Mobility Navigator ROS Node
    '''
    def __init__(self):
        super().__init__('x_mobility_navigator') #pycuda
        cuda.init() #pycuda
        self.ctx = cuda.Device(1).make_context()
        # Parameters
        self.declare_parameter(RUNTIME_PATH, '/tmp/x_mobility.engine') #model_path
        self.declare_parameter(MAPLESS_FLAG, True) #맵이 없는 상태에서 경로를 생성할지 여부 (기본 True)

        # Subscriber
        self.image_subscriber = self.create_subscription( 
            Image, IMAGE_TOPIC_NAME, self.image_callback, 10)#카메라 이미지(sensor_msgs/Image)를 구독하고 image_callback 함수로 처리
        self.odom_subscriber = self.create_subscription( 
            Odometry, ODOM_TOPIC_NAME, self.odom_callback, 10)#로봇 위치 및 속도 정보(nav_msgs/Odometry) 구독, odom_callback 호출
        self.route_subscriber = self.create_subscription( 
            Path, ROUTE_TOPIC_NAME, self.route_callback, 10)#경로 정보(nav_msgs/Path) 구독, route_callback 호출
        self.goal_subscriber = self.create_subscription( 
            PoseStamped, GOAL_TOPIC_NAME, self.goal_callback, 10)#최종 목표 위치(geometry_msgs/PoseStamped) 구독, goal_callback 호출

        # Publisher
        self.cmd_publisher = self.create_publisher(Twist, CMD_TOPIC_NAME, 10)
        self.path_publisher = self.create_publisher(Path, PATH_TOPIC_NAME, 10)

        # Timer
        self.timer = self.create_timer(0.2, self.inference) #0.2초(5Hz)마다 inference 함수 호출 (모델 추론 및 제어 반복 실행)

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Internal states
        self.action = np.zeros(6, dtype=np.float32)
        self.path = np.zeros(10, dtype=np.float32)
        self.history = np.zeros((1, 1024), dtype=np.float32)
        self.sample = np.zeros((1, 512), dtype=np.float32)
        self.camera_image = None # recent image data
        self.route_vectors = None
        self.goal = None
        self.ego_speed = None #robot speed
        self.runtime_context = None # TensorRT 추론 컨텍스트 객체
        self.stream = cuda.Stream() #  CUDA 비동기 처리용 스트림 객체
        self.cv_bridge = CvBridge()

    def load_model(self):
        #self.get_logger().info('Loading model')
        #runtime_path = self.get_parameter(
        #    RUNTIME_PATH).get_parameter_value().string_value
        #with open(runtime_path, "rb") as f:
        #    engine_data = f.read()

        # Create a TensorRT runtime
        #runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))# TensorRT 엔진 객체로 역직렬화(복원)
        #engine = runtime.deserialize_cuda_engine(engine_data)#실제 GPU에서 추론 가능한 실행 엔진 생성  TensorRT 엔진 객체로 역직렬화(복원)
        #self.runtime_context = engine.create_execution_context() #엔진에서 추론 실행 컨텍스트 생성 (실제로 추론할 때 사용)
                # 1) 컨텍스트를 명시적으로 활성화하고
        #self.ctx.push()
        self.get_logger().info('Loading model')
        runtime_path = self.get_parameter(RUNTIME_PATH).get_parameter_value().string_value
        with open(runtime_path, "rb") as f:
            engine_data = f.read()

        # runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))
        # engine  = runtime.deserialize_cuda_engine(engine_data)
        # self.runtime_context = engine.create_execution_context()

        runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))
        # 2) engine deserialize
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        # 3) inference context 생성
        self.runtime_context = self.engine.create_execution_context()

        # (디버그) 바인딩 정보 출력
        for i in range(self.engine.num_bindings):
            name  = self.engine.get_binding_name(i)
            shape = self.engine.get_binding_shape(i)
            io    = 'IN ' if self.engine.binding_is_input(i) else 'OUT'
            self.get_logger().info(f"[TRT] binding #{i}: {name}, shape={shape}, {io}")

        # 2) 로드가 끝나면 반드시 pop



    def image_callback(self, image_msg):
        self.camera_image = self.process_image_msg(image_msg) ##→ 수신한 이미지를 process_image_msg 함수로 처리하고 → 그 결과를 self.camera_image에 저장 (딥러닝 입력용)

    def odom_callback(self, odom_msg):
        self.ego_speed = np.array(odom_msg.twist.twist.linear.x,
                                  dtype=np.float32) # 로봇의 선속도(linear velocity) xx 방향 값을 추출하고 float32로 저장 로봇이 전진 중인지, 얼마나 빠른지를 추론에 반영하기 위해 사용

    def goal_callback(self, goal_msg): #목표 위치를 self.goal에 저장 모델 추론 시 사용되는 내부 상태 메모리 (history, sample) 초기화→ 새 목표가 주어졌으니 과거 상태는 버리고 처음부터 추론 시작
        self.goal = goal_msg
        # Reset history state
        self.history = np.zeros((1, 1024), dtype=np.float32) #    새로운 목표 위치에 맞춰 딥러닝 모델의 내부 상태도 초기화
        self.sample = np.zeros((1, 512), dtype=np.float32)

    def route_callback(self, route_msg):
        # Get transform between route and robot. 수신한 경로를 로봇 기준 좌표계로 변환(transform)하고, 벡터 배열을 생성해 저장
        try:
            transform = self.tf_buffer.lookup_transform(
                ROBOT_FRAME, route_msg.header.frame_id, Time())
                
        except TransformException as ex:
            self.get_logger().error(
                f'Could not transform {ROBOT_FRAME} to {route_msg.header.frame_id}: {ex}'
            )
            return
        
        route_poses = route_msg.poses
        num_poses = min(len(route_poses), NUM_ROUTE_POINTS)  
        # Return if route is empty.
        if num_poses == 0:
            return
        # Select the first NUM_ROUTE_POINTS and append the last route point as needed.
        indices = [idx for idx in range(num_poses)]
        indices.extend([num_poses - 1] * (NUM_ROUTE_POINTS - len(indices)))
        # Extract the x and y position in robot frame.
        selected_route_positions = []
    #현재 로봇 기준 좌표계(ROBOT_FRAME)로 경로를 변환 (TF 사용) -> 변환된 경로 중 최대 NUM_ROUTE_POINTS만큼 선택 -> 각 연속된 두 점을 이어서 [x1, y1, x2, y2] 형태의 route vector 배열을 생성 -> 최종적으로 self.route_vectors에 [19 x 4] 크기의 numpy 배열로 저장
        for idx in indices:
            transformed_pose = do_transform_pose(route_poses[idx].pose,
                                                 transform)
            selected_route_positions.append(
                [transformed_pose.position.x, transformed_pose.position.y])
        self.route_vectors = np.zeros(
            (NUM_ROUTE_POINTS - 1, ROUTE_VECTOR_SIZE), np.float32)
        for idx in range(NUM_ROUTE_POINTS - 1):
            self.route_vectors[idx] = np.concatenate(
                (selected_route_positions[idx],
                 selected_route_positions[idx + 1]),
                axis=0)

    def compose_mapless_route(self): #지도(map)나 외부 경로(Path) 없이도 로봇이 목표 지점까지 갈 수 있도록 내부적으로 경로(route)를 생성하는 함수
        if self.goal is None:
            return
        try:
            transform = self.tf_buffer.lookup_transform(
                ROBOT_FRAME, self.goal.header.frame_id, Time()) #목표 좌표를 로봇 기준 좌표계로 변환
        except TransformException as ex:
            self.get_logger().error(
                f'Could not transform {ROBOT_FRAME} to {self.goal.header.frame_id}: {ex}'
            )
            return
        goal_in_robot_frame = do_transform_pose(self.goal.pose, transform)
        route_poses = upsample_points(
            [0.0, 0.0],
            [goal_in_robot_frame.position.x, goal_in_robot_frame.position.y],
            1.0) # 시작점 ~ 목표점 사이를 일정 간격으로 보간 (upsample)
        num_poses = min(len(route_poses), NUM_ROUTE_POINTS) # 시작점 ~ 목표점 사이를 일정 간격으로 보간 (upsample)
        # Return if route is empty.
        if num_poses == 0:
            return
        # Select the first NUM_ROUTE_POINTS and append the last route point as needed.
        indices = [idx for idx in range(num_poses)]
        indices.extend([num_poses - 1] * (NUM_ROUTE_POINTS - len(indices))) #부족하면 마지막 점을 반복해서 채움
        # Extract the x and y position in robot frame.
        selected_route_positions = []
        for idx in indices:
            selected_route_positions.append(route_poses[idx]) #선택된 점들을 저장
        self.route_vectors = np.zeros(
            (NUM_ROUTE_POINTS - 1, ROUTE_VECTOR_SIZE), np.float32)
        for idx in range(NUM_ROUTE_POINTS - 1):
            self.route_vectors[idx] = np.concatenate(
                (selected_route_positions[idx],
                 selected_route_positions[idx + 1]),
                axis=0) #경로 벡터(route vector)로 변환

    # def inference(self): #딥러닝 모델을 실제로 실행(추론)하는 핵심 함수
    #     # Load model if not ready
    #     #self.ctx.push() #pycuda
    #     ##if not self.runtime_context:
    #     #    self.load_model()
    #     if not self.runtime_context:
    #         self.load_model()
    #     self.ctx.push()

    #     self.ctx.push()
    #     try:
    #         if not self.runtime_context:
    #             self.load_model()


    #     # Compose a simple route in mapless mode.
    #     if self.get_parameter(MAPLESS_FLAG).get_parameter_value().bool_value:
    #         self.compose_mapless_route()

    #     # Sanity checks of the inputs.
    #     # camera image, start-goal vectro. ego speed ok~
    #     # TODO: Sync the msgs.
    #     if self.camera_image is None or self.route_vectors is None or self.ego_speed is None:
    #         self.get_logger().info(f'Inputs are not ready.')
    #         #self.ctx.pop() #pycuda
    #         return

    #     self._trt_inference() #딥러닝 추론 실행
    #     self.publish_action()
    #     self.publish_path()
    #     #self.ctx.pop() #pycuda
    #     finally:
    #         self.ctx.pop()

    def inference(self):
        self.get_logger().info('[XMob] inference() called')
        self.ctx.push()
        try:
            if not self.runtime_context:
                self.get_logger().info('[XMob] loading model')
                self.load_model()

            if self.get_parameter(MAPLESS_FLAG).get_parameter_value().bool_value:
                self.get_logger().info('[XMob] composing mapless route')
                self.compose_mapless_route()

            if self.camera_image is None:
                self.get_logger().warn('[XMob] camera_image is None')
            if self.route_vectors is None:
                self.get_logger().warn('[XMob] route_vectors is None')
            if self.ego_speed is None:
                self.get_logger().warn('[XMob] ego_speed is None')

            if self.camera_image is None \
            or self.route_vectors is None \
            or self.ego_speed is None:
                self.get_logger().info('[XMob] Inputs are not ready.')
                return

            self.get_logger().info('[XMob] All inputs ready. Running inference')
            self._trt_inference()
            self.get_logger().info('[XMob] Inference done. Publishing...')
            self.publish_action()
            self.publish_path()

        finally:
            self.ctx.pop()

    def _trt_inference(self):
        # Allocate device memory for inputs.
        image_input = cuda.mem_alloc(self.camera_image.nbytes)
        route_vec_input = cuda.mem_alloc(self.route_vectors.nbytes)
        speed_input = cuda.mem_alloc(self.ego_speed.nbytes)
        action_input = cuda.mem_alloc(self.action.nbytes)
        history_input = cuda.mem_alloc(self.history.nbytes)
        sample_input = cuda.mem_alloc(self.sample.nbytes)
        action_output = cuda.mem_alloc(self.action.nbytes)
        path_output = cuda.mem_alloc(self.path.nbytes)
        history_output = cuda.mem_alloc(self.history.nbytes)
        sample_ouput = cuda.mem_alloc(self.sample.nbytes)

        # Copy inputs to device.
        cuda.memcpy_htod(image_input, self.camera_image)
        cuda.memcpy_htod(route_vec_input, self.route_vectors)
        cuda.memcpy_htod(speed_input, self.ego_speed)
        cuda.memcpy_htod(action_input, self.action)
        cuda.memcpy_htod(history_input, self.history)
        cuda.memcpy_htod(sample_input, self.sample) #CPU 메모리에 있는 넘파이 배열 → GPU 버퍼로 복사
        # Order bindings based on sequence encoded in engine
        # Run engine.get_binding_name(binding_idx) to verify
        bindings = [
            int(image_input),
            int(route_vec_input),
            int(speed_input),
            int(action_input),
            int(history_input),
            int(sample_input),
            int(action_output),
            int(path_output),
            int(history_output),
            int(sample_ouput),
        ]#바인딩 순서 정의,바인딩 순서는 모델이 export 될 때 정의된 순서와 정확히 같아야 함
        # Run inference
        self.runtime_context.execute_v2(bindings) #execute_v2()는 바인딩된 메모리로 추론 실행 -> 추론 결과는 GPU 메모리에 저장됨
        # Copy action back to host and publish
        cuda.memcpy_dtoh(self.action, action_output)
        cuda.memcpy_dtoh(self.path, path_output)
        cuda.memcpy_dtoh(self.history, history_output)
        cuda.memcpy_dtoh(self.sample, sample_ouput) #GPU 추론 결과를 다시 Python의 넘파이 배열로 복사



    def publish_action(self):
        cmd_vel = Twist()
        cmd_vel.linear.x = float(self.action[0])
        cmd_vel.angular.z = float(self.action[5])
        self.cmd_publisher.publish(cmd_vel)
 #TensorRT 추론 결과인 self.action 배열에서: [0]번 인덱스 → 로봇의 전진 속도 (linear x) [5]번 인덱스 → 로봇의 회전 속도 (angular z), 이를 ROS 2의 geometry_msgs/msg/Twist 메시지로 포장해서 /cmd_vel 토픽으로 퍼블리시함.

    def publish_path(self):
        path = Path()
        path.header.frame_id = ROBOT_FRAME
        path.header.stamp = self.get_clock().now().to_msg()
        for idx in range(len(self.route_vectors)):
            path_pose = PoseStamped() #경로의 한 점 생성
            path_pose.header = path.header #시간 & 좌표계 설정
            path_pose.pose.position.x = float(self.route_vectors[idx][0]) #그 점의 x, y 위치 지정 (로봇 기준 경로)
            path_pose.pose.position.y = float(self.route_vectors[idx][1])
            path.poses.append(path_pose)
        self.path_publisher.publish(path)
#    시각화 도구(Rviz)에서 실시간으로 로봇 경로



    # def process_image_msg(self, image_msg):
    #     image_channels = int(image_msg.step / image_msg.width)
    #     image = np.array(image_msg.data).reshape(
    #         (image_msg.height, image_msg.width, image_channels))
    #     image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
    #     return np.ascontiguousarray(image)
    # #OpenCV(RGB/HWC) → PyTorch/TensorRT(C×H×W) NumPy 배열 (TensorRT 입력용)
    def process_image_msg(self, image_msg):
        image_channels = int(image_msg.step / image_msg.width)
        image = np.array(image_msg.data).reshape(
            (image_msg.height, image_msg.width, image_channels))

        # 리사이즈: 엔진이 기대하는 해상도 (예: 1200x1920)
        image = cv2.resize(image, (1920, 1200), interpolation=cv2.INTER_LINEAR)

        # 전처리: HWC → CHW, 정규화, 배치 차원 추가
        image = image.astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)
        image = np.expand_dims(image, axis=0)

        return np.ascontiguousarray(image)



def main(args=None):
    rclpy.init(args=args)
    x_mobility_navigator = XMobilityNavigator()
    rclpy.spin(x_mobility_navigator)

#ROS 2 노드 초기화 & XMobilityNavigator 클래스 인스턴스 생성 & rclpy.spin() → 콜백 함수들이 계속 동작하도록 루프 유지 (ROS 2 이벤트 루프)


if __name__ == '__main__':
    main()
