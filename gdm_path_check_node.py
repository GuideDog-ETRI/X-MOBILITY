import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path as PathMsg
from rclpy.executors import MultiThreadedExecutor

class LocalPlanListener(Node):
    def __init__(self):
        super().__init__('gdm_path_check_node')
        # Subscribe to the /gdm/local_plan topic
        self.subscription = self.create_subscription(
            PathMsg,
            '/gdm/local_plan',
            self.listener_callback,
            10
        )
        self.get_logger().info('Node initialized and subscribed to /gdm/local_plan')

    def listener_callback(self, msg):
        if msg.poses:
            last_pose = msg.poses[-1].pose
            
            if (last_pose.position.x == 0.0 and last_pose.position.y == 0.0 and last_pose.position.z == 0.0 and
                last_pose.orientation.x == 0.0 and last_pose.orientation.y == 0.0 and last_pose.orientation.z == 0.0 and last_pose.orientation.w == 1.0):
                
                ## 다른 GP 수신이 필요합니다.
                self.get_logger().info('Please provide another GP.')

            else:
                self.get_logger().info(f'Generated path for position (x={last_pose.position.x}, y={last_pose.position.y}, z={last_pose.position.z}), '
                    f'Orientation (x={last_pose.orientation.x}, y={last_pose.orientation.y}, z={last_pose.orientation.z}, w={last_pose.orientation.w})')
            
        
        else:
            self.get_logger().info('No poses found in the Path message.')



def main(args=None):
    rclpy.init(args=args)
    
    path_check_node = LocalPlanListener()
    executor = MultiThreadedExecutor()
    executor.add_node(path_check_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        executor.get_logger().info('Node stopped cleanly.')
    except Exception as e:
        executor.get_logger().error(f'Error occurred: {e}')
    finally:
        path_check_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
