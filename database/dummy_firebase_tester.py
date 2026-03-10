import firebase_admin
from firebase_admin import credentials, db
import time
import random

# 1. Firebase 인증 키 연동 (경로를 연구원님의 키 파일에 맞게 수정하세요)
# 예: 'database/serviceAccountKey.json'
cred = credentials.Certificate("serviceAccountKey.json")

# 2. 본인의 Firebase Realtime Database URL을 입력하세요.
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://rehab-aa1ee-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

# 3. 테스트할 경로 설정 (UI가 lateral_raise_sessions를 감시 중입니다)
ref = db.reference('lateral_raise_sessions/test_dummy_session')

print("🚀 가상 데이터 전송을 시작합니다... (3초마다 횟수 1 증가)")

rep_count = 0

try:
    while True:
        # UI가 에러 없이 읽을 수 있도록 전체 JSON 구조를 쏴줍니다.
        dummy_data = {
            "exercise_type": "lateral_raise",
            "rep_count": rep_count,
            "last_feedback": f"{rep_count}회 성공! 완벽한 자세입니다!",
            "good_posture_ratio": random.randint(80, 100),
            "robot_assist_parameters": {
                "target_prom": 90, "assist_trigger_angle": 45, "pure_arom": 50
            },
            "elderly_pt_metrics": {
                "avg_successful_peak_angle": 85.5,
                "avg_all_peak_angle": 80.0,
                "max_rom_left": random.randint(80, 90),
                "max_rom_right": random.randint(80, 90),
                "avg_rep_duration_sec": 3.2,
                "tremor_count": 0,
                "max_z_depth_drift_mm": 12
            },
            "warning_counts": {
                "arm_balance_issue": 0, "arms_too_high": 0, "lean_back_momentum": 0, "chest_down": 0
            },
            "realtime_joints": {
                # 차트가 움직이는 걸 보기 위해 랜덤 각도 생성
                "left_shoulder": random.randint(20, 90),
                "right_shoulder": random.randint(20, 90)
            }
        }

        # DB 업데이트 수행
        ref.set(dummy_data)
        print(f"✅ 데이터 전송 완료! 현재 횟수: {rep_count}")
        
        rep_count += 1
        time.sleep(3) # 3초 대기 (애니메이션 볼 시간 확보)

except KeyboardInterrupt:
    print("\n⏹️ 테스트를 종료합니다.")