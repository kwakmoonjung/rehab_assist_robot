import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # 현재 파이썬 파일이 있는 폴더 경로를 자동으로 찾아서 csv 파일과 연결합니다.
    csv_file = "/home/rokey/cobot_ws/src/cobot2_ws/rehab_assist_robot/object_detection/pose_metrics.csv"

    if not os.path.exists(csv_file):
        print(f"❌ 에러: {csv_file} 파일을 찾을 수 없습니다.")
        return

    # 데이터 불러오기
    df = pd.read_csv(csv_file)
    
    # 타임스탬프를 0초부터 시작하도록 정규화
    df['Time_s'] = df['Timestamp'] - df['Timestamp'].iloc[0]

    # 그래프 스타일 및 크기 설정
    plt.style.use('ggplot')
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle('MediaPipe Performance Benchmark (Elbow ROM)', fontsize=16, fontweight='bold')

    # ---------------------------------------------------
    # 1. 각도(Angle) 추적 정확도 및 90도 타겟 확인
    # ---------------------------------------------------
    axs[0].plot(df['Time_s'], df['Angle'], color='dodgerblue', linewidth=2, label='Measured Angle')
    axs[0].axhline(y=90, color='red', linestyle='--', linewidth=2, label='Ground Truth (90° Target)')
    axs[0].set_title("1. Elbow Angle Tracking (Accuracy & ROM)", fontsize=13)
    axs[0].set_ylabel("Angle (Degrees)")
    axs[0].legend(loc='upper right')
    axs[0].set_ylim(0, 190)

    # ---------------------------------------------------
    # 2. 실시간성(FPS) 방어율 확인
    # ---------------------------------------------------
    avg_fps = df['FPS'].mean()
    axs[1].plot(df['Time_s'], df['FPS'], color='mediumseagreen', linewidth=2, label=f'FPS (Avg: {avg_fps:.1f})')
    axs[1].axhline(y=avg_fps, color='darkgreen', linestyle=':', linewidth=2, label='Average FPS')
    axs[1].set_title("2. Real-time Performance (FPS Stability)", fontsize=13)
    axs[1].set_ylabel("Frames Per Second")
    axs[1].legend(loc='lower right')

    # ---------------------------------------------------
    # 3. 관절 미세 떨림(Jitter) 수치 확인
    # ---------------------------------------------------
    avg_jitter = df['Jitter'].mean()
    axs[2].plot(df['Time_s'], df['Jitter'], color='tomato', linewidth=1.5, label=f'Jitter (Avg: {avg_jitter:.2f}°)')
    axs[2].set_title("3. Signal Stability (Angle Jitter)", fontsize=13)
    axs[2].set_ylabel("Jitter (Degrees)")
    axs[2].set_xlabel("Time (s)")
    axs[2].legend(loc='upper right')

    # 화면에 출력
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

if __name__ == '__main__':
    main()