import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm

def main():
    # --- [한글 폰트 설정] ---
    font_path = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rc('font', family='NanumBarunGothic')
    plt.rcParams['axes.unicode_minus'] = False

    # [수정] 성적표 경로 리스트 (YOLO11n, YOLO11s만 남김)
    paths = {
        "YOLO11n": "/tmp/yolo11n_metrics.csv",
        "YOLO11s": "/tmp/yolo11s_metrics.csv"
    }

    # 데이터 로드 및 정규화
    data = {}
    for name, path in paths.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            # 시간 축 정규화 (0초부터 시작)
            df['Time_s'] = df['Timestamp'] - df['Timestamp'].iloc[0]
            data[name] = df
        else:
            print(f"{name} 데이터가 없습니다. ({path})")

    if not data:
        print("비교할 데이터가 없습니다.")
        return

    plt.style.use('ggplot')
    fig, axs = plt.subplots(3, 1, figsize=(12, 16))
    
    # [수정] 그래프 제목 변경
    fig.suptitle('AI Pose Estimation Benchmark: YOLO11n vs YOLO11s', fontsize=16, fontweight='bold')

    # [수정] 색상 지정 (YOLO11n, YOLO11s만 남김)
    colors = {"YOLO11n": "green", "YOLO11s": "magenta"}

    # [1] Elbow Angle Tracking
    for name, df in data.items():
        # [수정] 선 굵기 동일하게 설정
        lw = 2.0 
        axs[0].plot(df['Time_s'], df['Angle'], label=name, color=colors[name], alpha=0.8, linewidth=lw)
    axs[0].axhline(y=90, color='black', linestyle='--', label='Target (90°)')
    axs[0].set_title("1. Elbow Angle Tracking (정확도 및 가동 범위)")
    axs[0].set_ylabel("Angle (deg)")
    axs[0].set_ylim(0, 190)
    axs[0].legend(loc='upper right')

    # [2] Real-time Performance
    for name, df in data.items():
        axs[1].plot(df['Time_s'], df['FPS'], label=f"{name} (Avg: {df['FPS'].mean():.1f})", color=colors[name], alpha=0.7)
    axs[1].set_title("2. Real-time Performance (실시간성 / FPS)")
    axs[1].set_ylabel("FPS")
    axs[1].legend(loc='lower right')

    # [3] Jitter Analysis 
    for name, df in data.items():
        # 각도의 변화량(Jitter) 계산
        jitter = df['Angle'].diff().abs().fillna(0)
        axs[2].plot(df['Time_s'], jitter, label=f"{name} Jitter (Avg: {jitter.mean():.2f}°)", color=colors[name], alpha=0.5)
    axs[2].set_title("3. Signal Stability (각도 떨림 / Jitter)")
    axs[2].set_ylabel("Jitter (deg)")
    axs[2].set_xlabel("Time (s)")
    axs[2].legend(loc='upper right')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

if __name__ == '__main__':
    main()