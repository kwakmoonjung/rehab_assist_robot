import pandas as pd
import glob
import os

def main():
    # 1. 현재 폴더 내의 모든 metrics.csv 파일 탐색
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_files = glob.glob(os.path.join(current_dir, '*_metrics.csv'))
    
    if not csv_files:
        print("비교할 데이터(_metrics.csv)가 존재하지 않습니다.")
        return

    results = []
    
    for file in csv_files:
        model_name = os.path.basename(file).replace('_metrics.csv', '').upper()
        try:
            df = pd.read_csv(file)
            if df.empty:
                continue
                
            # [지표 1] Real-time Performance (실시간성 / Avg FPS)
            avg_fps = df['FPS'].mean()
            
            # [지표 2] Elbow Angle Tracking (정확도 및 가동범위 / Min ~ Max Angle)
            min_angle = df['Angle'].min()
            max_angle = df['Angle'].max()
            
            # [지표 3] Signal Stability (각도 떨림 / Jitter)
            jitter = df['Angle'].diff().abs().mean()
            
            # 추적 실패 여부 판별 (관절 인식을 놓쳐 0도 근처로 튀었는지 확인)
            is_failed = True if min_angle < 10 else False
            
            results.append({
                '모델명': model_name,
                'Elbow Angle Tracking (Min~Max, °)': f"{min_angle:.1f} ~ {max_angle:.1f}",
                'Real-time Performance (Avg FPS)': round(avg_fps, 1),
                'Signal Stability (Jitter, °)': round(jitter, 2),
                '_min_angle': min_angle,
                '_fps': avg_fps,
                '_jitter': jitter,
                '_is_failed': is_failed
            })
        except Exception as e:
            print(f"{model_name} 파일 분석 중 에러 발생: {e}")

    if not results:
        return

    df_summary = pd.DataFrame(results)
    
    # 2. 출력용 표 정렬 (Jitter가 낮고 FPS가 높은 순)
    df_summary = df_summary.sort_values(by=['_jitter', '_fps'], ascending=[True, False]).reset_index(drop=True)
    
    display_cols = ['모델명', 'Elbow Angle Tracking (Min~Max, °)', 'Real-time Performance (Avg FPS)', 'Signal Stability (Jitter, °)']
    df_display = df_summary[display_cols].copy()
    
    # 터미널 표 출력
    print("\n" + "="*85)
    print(" 📊 AI Pose Estimation 3대 핵심 지표 종합 비교표")
    print("="*85)
    
    # tabulate 패키지가 없을 경우를 대비해 예외 처리 추가
    try:
        print(df_display.to_markdown(index=False))
    except ImportError:
        print(df_display.to_string(index=False))
        
    print("="*85 + "\n")

    # [추가] 비교 결과 표를 CSV 파일로 저장
    save_path = os.path.join(current_dir, 'model_table.csv')
    df_display.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"✅ 비교 결과 표가 저장되었습니다: {save_path}\n")

    # --- 3. 최종 모델 자동 선택 알고리즘 ---
    print("🏆 [알고리즘 기반 최종 모델 선정 결과]")
    
    # Step 1. 치명적 오류 필터링 (가동범위 추적 실패: Min Angle < 10도 제거)
    valid_models = df_summary[~df_summary['_is_failed']].copy()
    
    if valid_models.empty:
        print("⚠️ 모든 모델이 추적에 실패한 구간이 있습니다. 가장 Jitter가 낮은 모델을 선택합니다.")
        best_model = df_summary.loc[df_summary['_jitter'].idxmin()]
    else:
        # Step 2. 실시간성 필터링 (로봇 제어 마지노선: 15 FPS 이상)
        realtime_models = valid_models[valid_models['_fps'] >= 15.0].copy()
        
        if realtime_models.empty:
            print("⚠️ 실시간 제어 기준(15 FPS)을 만족하는 안정적인 모델이 없습니다. 가동범위가 정상인 것 중 가장 빠른 모델을 선택합니다.")
            best_model = valid_models.loc[valid_models['_fps'].idxmax()]
        else:
            # Step 3. 최종 선택: 기준을 통과한 엘리트 모델 중 가장 안정적인(Jitter가 제일 낮은) 모델 선택
            best_model = realtime_models.loc[realtime_models['_jitter'].idxmin()]
    
    best_name = best_model['모델명']
    print(f"👉 시스템이 분석한 재활 로봇 제어 최적의 모델은 **{best_name}** 입니다!\n")
    print("💡 [선정 이유 요약]")
    
    if best_model['_is_failed']:
        print("- 완벽한 모델은 없으나, 후보군 중 신호 떨림이 가장 적어 제어 오작동 확률이 가장 낮습니다.")
    else:
        print("1. Elbow Angle Tracking: 치명적인 인식 실패(관절 각도 0도 수렴 현상)가 전혀 발생하지 않았습니다.")
        print(f"2. Real-time Performance: 초당 {best_model['_fps']:.1f} 프레임으로, 로봇과 실시간 연동 제어에 충분한 속도를 보장합니다.")
        print(f"3. Signal Stability: 후보군 중 각도 떨림(Jitter: {best_model['_jitter']:.2f}°)이 가장 낮아, 환자에게 부드러운 로봇 모터 구동을 제공할 수 있습니다.")
        
    print("="*85 + "\n")

if __name__ == '__main__':
    main()