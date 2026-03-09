import os
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class ExerciseLoggerNode(Node):
    def __init__(self):
        super().__init__('exercise_logger_node')
        
        # 저장할 로컬 경로 (임시 JSON 저장용, 추후 DB 쿼리문으로 변경될 곳)
        self.log_file = os.path.expanduser("~/exercise_session_log.json")
        
        # /exercise_result 토픽 구독
        self.subscription = self.create_subscription(
            String,
            '/exercise_result',
            self.result_callback,
            10
        )
        self.get_logger().info("💾 데이터베이스 로거 노드가 시작되었습니다. 토픽 수신 대기 중...")
        self.get_logger().info(f"현재 타겟 파일: {self.log_file}")

    def result_callback(self, msg):
        try:
            # 수신받은 JSON 문자열을 파이썬 딕셔너리로 변환
            data = json.loads(msg.data)
            
            # 추후 이 자리에 INSERT INTO ... 같은 DB 쿼리가 들어갑니다.
            # 지금은 우선 파일로 저장합니다.
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            self.get_logger().debug(f"데이터 갱신 완료 (Count: {data.get('rep_count')})")
            
        except json.JSONDecodeError as e:
            self.get_logger().error(f"JSON 파싱 에러: {e}")
        except Exception as e:
            self.get_logger().error(f"데이터 저장 중 에러 발생: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ExerciseLoggerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()