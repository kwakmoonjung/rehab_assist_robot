let angleChart = null; 

document.addEventListener("DOMContentLoaded", function() {
    document.getElementById('btn_toggle_admin').addEventListener('click', function() {
        const panel = document.getElementById('admin_panel');
        if (panel.style.display === 'none') {
            panel.style.display = 'block';
            this.classList.replace('btn-outline-secondary', 'btn-secondary'); 
        } else {
            panel.style.display = 'none';
            this.classList.replace('btn-secondary', 'btn-outline-secondary');
        }
    });

    document.getElementById('btn_start_exercise').addEventListener('click', function() {
        const selector = document.getElementById('exercise_selector');
        const exerciseValue = selector.value;
        if (!exerciseValue) { alert("진행하실 운동을 먼저 선택해주세요!"); return; }
        const exerciseText = selector.options[selector.selectedIndex].text;

        UIManager.startExerciseUI(exerciseValue, exerciseText);
        
        this.classList.replace('btn-primary', 'btn-success');
        this.innerHTML = '<i class="fas fa-check"></i> 시작됨';
        setTimeout(() => {
            this.classList.replace('btn-success', 'btn-primary');
            this.innerHTML = '<i class="fas fa-play"></i> 운동 시작';
        }, 1500);
    });

    document.getElementById('btn_stop_exercise').addEventListener('click', function() {
        document.getElementById('sys_subtitle').innerText = "운동이 중지되었습니다. 다음 운동을 선택해주세요.";
        const feedbackEl = document.getElementById('main_feedback');
        feedbackEl.innerText = "운동 정지 완료.";
        feedbackEl.style.backgroundColor = "#ffebee";
        feedbackEl.style.color = "#c62828";
    });

    document.getElementById('btn_reset_exercise').addEventListener('click', function() {
        UIManager.resetUI();
    });
});

const UIManager = {
    currentExercise: 'lateral_raise', 

    resetUI: function() {
        document.getElementById('exercise_selector').selectedIndex = 0;
        document.getElementById('sys_subtitle').innerText = "대기 중... 운동을 선택해주세요.";
        const feedbackEl = document.getElementById('main_feedback');
        feedbackEl.innerText = '"대기 중입니다..."';
        feedbackEl.style.backgroundColor = "#e3f2fd";
        feedbackEl.style.color = "#0d47a1";
        document.getElementById('chart_title').innerHTML = '<i class="fas fa-chart-line text-info"></i> 실시간 관절 궤적 (대기 중)';
        document.getElementById('rep_count_main').innerText = 0;
        document.getElementById('main_max_rom_left').innerText = "0";
        document.getElementById('main_max_rom_right').innerText = "0";
        for(let i=1; i<=7; i++) {
            let el = document.getElementById(`metric_val_${i}`);
            if(el) el.innerText = "0";
        }
        for(let i=1; i<=4; i++) {
            let warn = document.getElementById(`warn_val_${i}`);
            if(warn) warn.innerText = "0";
        }
        if (angleChart) {
            angleChart.data.labels = [];
            angleChart.data.datasets[0].data = [];
            angleChart.data.datasets[1].data = [];
            angleChart.update('none');
        }
    },

    // 🌟 1. 종목별 UI 라벨 (글씨) 변경
    startExerciseUI: function(exerciseValue, exerciseName) {
        this.currentExercise = exerciseValue; 
        document.getElementById('sys_subtitle').innerText = `${exerciseName} 실시간 모니터링 시스템`;
        
        if (exerciseValue === 'lateral_raise') {
            document.getElementById('chart_title').innerHTML = '<i class="fas fa-chart-line text-info"></i> 실시간 관절 궤적 (L: 5-7-11 / R: 6-8-12)';
            document.getElementById('main_metric_label_left').innerText = "좌측 최고 도달";
            document.getElementById('main_metric_label_right').innerText = "우측 최고 도달";
            
            // 지표 라벨
            document.getElementById('metric_label_1').innerText = "완벽 자세 평균 도달";
            document.getElementById('metric_label_2').innerText = "전체 시도 평균 도달";
            document.getElementById('metric_label_3').innerText = "좌측 최고";
            document.getElementById('metric_label_4').innerText = "우측 최고";
            document.getElementById('metric_label_5').innerText = "운동 템포";
            document.getElementById('metric_label_6').innerText = "미세 떨림 감지";
            document.getElementById('metric_label_7').innerText = "앞쏠림(Z축) 편차";
            // 경고 라벨
            document.getElementById('warn_label_1').innerText = "양팔 밸런스 붕괴";
            document.getElementById('warn_label_2').innerText = "팔 과도하게 올림";
            document.getElementById('warn_label_3').innerText = "허리 반동 사용";
            document.getElementById('warn_label_4').innerText = "상체 숙여짐";

        } else if (exerciseValue === 'shoulder_press') {
            document.getElementById('chart_title').innerHTML = '<i class="fas fa-chart-line text-warning"></i> 실시간 관절 궤적 (L: 5-7-9 / R: 6-8-10)';
            document.getElementById('main_metric_label_left').innerText = "어깨 평균 각도";
            document.getElementById('main_metric_label_right').innerText = "팔꿈치 평균 각도";
            
            // 지표 라벨
            document.getElementById('metric_label_1').innerText = "평균 어깨 각도";
            document.getElementById('metric_label_2').innerText = "평균 팔꿈치 각도";
            document.getElementById('metric_label_3').innerText = "평균 몸통 각도";
            document.getElementById('metric_label_4').innerText = "전체 프레임";
            document.getElementById('metric_label_5').innerText = "정상 자세 프레임";
            document.getElementById('metric_label_6').innerText = "미사용 지표";
            document.getElementById('metric_label_7').innerText = "미사용 지표";
            // 경고 라벨
            document.getElementById('warn_label_1').innerText = "양팔 밸런스 붕괴";
            document.getElementById('warn_label_2').innerText = "상체 정렬 무너짐"; 
            document.getElementById('warn_label_3').innerText = "팔 너무 깊게 내림"; 
            document.getElementById('warn_label_4').innerText = "최하단 긴장 풀림"; 
        }

        const feedbackEl = document.getElementById('main_feedback');
        feedbackEl.innerText = "운동이 시작되었습니다. 준비 자세를 취해주세요.";
        feedbackEl.style.backgroundColor = "#e8f5e9"; 
        feedbackEl.style.color = "#2e7d32";
        
        document.getElementById('rep_count_main').innerText = 0;
    },

    // 🌟 2. DB에서 들어온 Raw Data를 종목에 맞게 분해하여 UI에 뿌리기
    updateDashboard: function(data) {
        if(!data) return;
        const exType = data.exercise_type || this.currentExercise;

        // 공통 데이터 갱신
        if(data.rep_count !== undefined) document.getElementById('rep_count_main').innerText = data.rep_count;
        if(data.last_feedback) {
            document.getElementById('last_feedback').innerText = data.last_feedback;
            document.getElementById('main_feedback').innerText = data.last_feedback;
        }

        let goodRatio = data.good_posture_ratio !== undefined ? data.good_posture_ratio : (data.performance_stats?.good_posture_ratio || 0);
        document.getElementById('good_posture_ratio').innerText = goodRatio;

        // 종목별 데이터 분기 처리 
        if (exType === 'lateral_raise') {
            // [사레레 JSON 구조 파싱]
            let params = data.robot_assist_parameters || {};
            document.getElementById('target_prom').innerText = params.target_prom || 0;
            document.getElementById('assist_trigger_angle').innerText = params.assist_trigger_angle || 0;
            document.getElementById('pure_arom').innerText = params.pure_arom || 0;

            let metrics = data.elderly_pt_metrics || {};
            document.getElementById('metric_val_1').innerText = metrics.avg_successful_peak_angle || 0;
            document.getElementById('metric_val_2').innerText = metrics.avg_all_peak_angle || 0;
            document.getElementById('metric_val_3').innerText = (metrics.max_rom_left || 0) + "°";
            document.getElementById('metric_val_4').innerText = (metrics.max_rom_right || 0) + "°";
            document.getElementById('metric_val_5').innerText = (metrics.avg_rep_duration_sec || 0) + "s";
            document.getElementById('metric_val_6').innerText = (metrics.tremor_count || 0) + "회";
            document.getElementById('metric_val_7').innerText = (metrics.max_z_depth_drift_mm || 0) + "mm";

            document.getElementById('main_max_rom_left').innerText = metrics.max_rom_left || 0;
            document.getElementById('main_max_rom_right').innerText = metrics.max_rom_right || 0;

            let warns = data.warning_counts || {};
            document.getElementById('warn_val_1').innerText = warns.arm_balance_issue || 0;
            document.getElementById('warn_val_2').innerText = warns.arms_too_high || 0;
            document.getElementById('warn_val_3').innerText = warns.lean_back_momentum || 0;
            document.getElementById('warn_val_4').innerText = warns.chest_down || 0;

        } else if (exType === 'shoulder_press') {
            // [숄더 프레스 JSON 구조 파싱]
            document.getElementById('target_prom').innerText = "-"; // 로봇 파라미터 없음
            document.getElementById('assist_trigger_angle').innerText = "-";
            document.getElementById('pure_arom').innerText = "-";

            document.getElementById('metric_val_1').innerText = data.avg_shoulder_angle || 0;
            document.getElementById('metric_val_2').innerText = data.avg_elbow_angle || 0;
            document.getElementById('metric_val_3').innerText = (data.avg_trunk_angle || 0) + "°";
            document.getElementById('metric_val_4').innerText = (data.frame_count || 0) + "프레임";
            document.getElementById('metric_val_5').innerText = (data.good_frame_count || 0) + "프레임";
            document.getElementById('metric_val_6').innerText = "-";
            document.getElementById('metric_val_7').innerText = "-";

            document.getElementById('main_max_rom_left').innerText = data.avg_shoulder_angle || 0;
            document.getElementById('main_max_rom_right').innerText = data.avg_elbow_angle || 0;

            let warns = data.warning_counts || {};
            document.getElementById('warn_val_1').innerText = warns.arm_balance_issue || 0;
            document.getElementById('warn_val_2').innerText = warns.body_not_straight || 0;
            document.getElementById('warn_val_3').innerText = warns.too_low || 0;
            document.getElementById('warn_val_4').innerText = warns.bend_elbows_at_bottom || 0;
        }

        // 실시간 차트 업데이트 (숄더프레스 JSON에 realtime_joints가 없으면 에러 안 나게 방어 로직)
        if(data.realtime_joints) {
            this.updateRealtimeChart(data.realtime_joints.left_shoulder, data.realtime_joints.right_shoulder);
        }
    },

    updateConnectionStatus: function(isOnline) {
        const statusEl = document.getElementById('connection-status');
        if (isOnline) {
            statusEl.innerHTML = '<i class="fas fa-wifi"></i> 실시간 연동 중';
            statusEl.className = "badge bg-success p-2 fs-6 me-3";
        } else {
            statusEl.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 연결 대기 중...';
            statusEl.className = "badge bg-secondary p-2 fs-6 me-3";
        }
    },

    initChart: function() {
        const ctx = document.getElementById('jointAngleChart').getContext('2d');
        angleChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [], 
                datasets: [
                    { label: '좌측', data: [], borderColor: '#198754', backgroundColor: 'rgba(25, 135, 84, 0.1)', borderWidth: 2, tension: 0.4, pointRadius: 0 },
                    { label: '우측', data: [], borderColor: '#0d6efd', backgroundColor: 'rgba(13, 110, 253, 0.1)', borderWidth: 2, tension: 0.4, pointRadius: 0 }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: { 
                    y: { min: 0, max: 180, title: { display: true, text: 'Angle (Deg)' } },
                    x: { display: true, title: { display: false } }
                },
                plugins: { legend: { position: 'top' } },
                animation: false 
            }
        });
    },

    updateRealtimeChart: function(leftAngle, rightAngle) {
        if (!angleChart) { this.initChart(); }
        const now = new Date().toLocaleTimeString('ko-KR', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
        angleChart.data.labels.push(now);
        angleChart.data.datasets[0].data.push(leftAngle);
        angleChart.data.datasets[1].data.push(rightAngle);
        if (angleChart.data.labels.length > 50) {
            angleChart.data.labels.shift(); angleChart.data.datasets[0].data.shift(); angleChart.data.datasets[1].data.shift();
        }
        angleChart.update('none'); 
    }
};