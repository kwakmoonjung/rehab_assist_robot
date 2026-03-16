# 🤖 재활 보조 및 운동 자세 교정 로봇 (Rehab Assist Robot)

이 프로젝트는 ROS2 Humble 환경에서 구동되는 **두산 M0609 협동 로봇 기반의 재활 및 운동 보조 시스템**입니다.
YOLOv11을 활용한 실시간 자세 인식(Pose Tracking)과 안면 인식 기능을 통해 사용자를 식별하고, 사레레(사이드 레터럴 레이즈), 숄더프레스 등의 운동 자세를 분석하여 로봇이 알맞은 교정 동작과 음성 안내를 제공합니다.

---

## 📌 주요 기능 (Key Features)

**1. 실시간 자세 인식 및 운동 교정 (Pose Tracking & Correction)**
* **탐지:** `yolo11n-pose.pt` 모델을 활용하여 사용자의 관절 포인트를 실시간으로 추적합니다.
* **평가:** 사레레, 숄더프레스 등의 운동 수행 시 올바른 궤적과 자세인지 분석합니다.
* **제어:** 자세 교정 완료 시 음성 안내와 함께 로봇이 초기 위치(init_pos)로 안전하게 복귀합니다.

**2. 사용자 맞춤형 안면 인식 및 DB 연동 (Face Recognition & DB)**
* **탐지:** 프로그램 시작 시 사용자 얼굴을 인식하여 개인 프로필을 식별합니다.
* **관리:** 2차 통합 테스트가 완료된 DB 및 UI 노드를 통해 개인별 운동 기록과 진행 상태를 저장하고 시각화합니다.

**3. 음성 처리 및 상호작용 (Voice Processing)**
* 운동 피드백과 로봇의 현재 상태를 사용자에게 음성으로 직관적으로 안내합니다.

---

## 🛠️ 시스템 설계 (System Architecture)

**전체 구조**
시스템은 크게 Perception(인식), Decision(판단/DB), Control(제어/안내) 세 파트로 구성됩니다.

1.  **Perception:** 카메라 센서를 통해 들어온 이미지 데이터를 바탕으로 얼굴 인식을 수행하고, YOLOv11 Pose 모델로 관절 데이터를 추출합니다.
2.  **Decision:** 추출된 자세 데이터를 분석하여 현재 운동의 정확도를 판별하고, DB와 연동하여 사용자 UI에 상태를 업데이트합니다.
3.  **Control:** ROS2 로봇 제어 노드를 통해 두산 M0609 로봇 팔의 모션을 제어(자세 가이드 및 초기화)하며, `voice_processing` 노드를 통해 피드백을 출력합니다.

---

## 🔄 알고리즘 플로우 차트 (Logic Flow)

![알고리즘 플로우 차트](rehab_assist_robot_flowchart-Flow_chart_final.drawio.png)
---

## 💻 개발 환경 (Environment)

* **OS:** Ubuntu 22.04 LTS (Jammy Jellyfish)
* **Middleware:** ROS 2 Humble Hawksbill
* **Language:** Python 3.10
* **Key Libraries:** `rclpy`, `ultralytics` (YOLO), `opencv-python`

---

## ⚙️ 사용 장비 (Hardware Setup)

본 프로젝트는 노인 재활 보조를 위한 안전한 환경 구축을 목표로 개발되었습니다.

| Component | Type | Spec / Description |
| :--- | :--- | :--- |
| **Robot** | Doosan M0609 | 6축 협동 로봇 (가반하중 6kg) |
| **Vision** | Web Camera | 2D RGB Camera (안면 및 자세 인식용) |
| **Compute** | PC | ROS2 메인 컨트롤 및 딥러닝 추론 |

---

## 🚀 설치 및 실행 순서 (Installation & How to Run)

프로젝트 구동을 위한 필수 라이브러리 설치, 워크스페이스 빌드, 그리고 전체 노드 실행 과정입니다. 아래 명령어들을 순서대로 실행해 주세요.

```bash
# 1. Python 필수 라이브러리 설치
pip install ultralytics opencv-python numpy

# 2. 프로젝트 워크스페이스 빌드
cd ~/ros2_ws
colcon build --packages-select rehab_assist_robot robot_control voice_processing object_detection

# 3. ROS2 환경 설정 및 메인 런치 파일 실행 (터미널 1)
ros_set
ros2 launch rehab_assist_robot main_system.launch.py

# 4. 객체/자세 인식 및 제어 노드 실행 (새 터미널 2를 열고 실행)
ros_set
ros2 run object_detection pose_tracking_node
