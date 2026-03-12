import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger

class SystemController(Node):
    def __init__(self):
        super().__init__('system_controller_node')

        # 1. 음성 비서로부터 시스템 명령을 받는 Subscriber
        self.cmd_sub = self.create_subscription(
            String,
            '/system_command',
            self.command_callback,
            10
        )

        # 2. 비전 노드(pose_analyzer)를 제어하기 위한 Service Client 생성
        self.cli_set_exercise = self.create_client(SetBool, '/set_exercise_state')
        self.cli_publish_3d = self.create_client(Trigger, '/publish_target_3d')

        self.get_logger().info("🎯 시스템 컨트롤러 노드 시작! 각 노드의 대기를 확인하고 명령을 기다립니다.")

    def command_callback(self, msg):
        """음성 비서에서 퍼블리시한 문자열 명령을 받아 분기 처리"""
        command = msg.data
        self.get_logger().info(f"수신된 명령: {command}")

        if command == "START_EXERCISE":
            self.get_logger().info("'운동 시작' 명령 확인 -> 비전 노드 로깅 On 요청")
            self.call_set_exercise_state(True)
            
        elif command == "END_EXERCISE": # [추가] 운동 명시적 종료
            self.get_logger().info("'운동 종료' 명령 확인 -> 비전 노드 로깅 Off 요청")
            self.call_set_exercise_state(False)
            
        elif command == "REPORT_EXERCISE": # [수정] 상태 변경 로직 제거, 단순 로깅 확인용
            self.get_logger().info("'운동 기록 조회' 명령 확인 -> 상태 유지")
        
        elif command == "TODAY_ROUTINE":
            self.get_logger().info("'오늘 루틴 추천' 명령 확인 -> 상태 유지")
            
        elif command == "CORRECTION":
            self.get_logger().info("'자세 교정' 명령 확인 -> 비전 노드에 3D 좌표 발행 요청")
            self.call_publish_target_3d()
            
        else:
            self.get_logger().warn(f"알 수 없는 명령입니다: {command}")

    def call_set_exercise_state(self, state):
        """비전 노드의 운동 상태 On/Off 서비스 호출"""
        if not self.cli_set_exercise.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("❌ '/set_exercise_state' 서비스가 응답하지 않습니다. (pose_analyzer 실행 확인 요망)")
            return
            
        req = SetBool.Request()
        req.data = state
        future = self.cli_set_exercise.call_async(req)
        future.add_done_callback(self.set_exercise_done_callback)

    def set_exercise_done_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info(f"✅ 비전 노드 응답: {response.message}")
        except Exception as e:
            self.get_logger().error(f"서비스 호출 실패: {e}")

    def call_publish_target_3d(self):
        """비전 노드에 3D 교정 좌표 발행 서비스 호출"""
        if not self.cli_publish_3d.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("❌ '/publish_target_3d' 서비스가 응답하지 않습니다. (pose_analyzer 실행 확인 요망)")
            return
            
        req = Trigger.Request()
        future = self.cli_publish_3d.call_async(req)
        future.add_done_callback(self.publish_3d_done_callback)

    def publish_3d_done_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info(f"✅ 비전 노드 응답: {response.message}")
        except Exception as e:
            self.get_logger().error(f"서비스 호출 실패: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = SystemController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()