/* ui_manager.js */
let angleChart = null; 
let reportChart = null; 
let sessionTimerInterval = null;
let sessionSeconds = 0;

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

    const reportModalEl = document.getElementById('reportModal');
    if(reportModalEl) {
        reportModalEl.addEventListener('show.bs.modal', function () {
            UIManager.generateGrowthReport();
        });
    }

    UIManager.initChart();
});

const UIManager = {
    currentExercise: '', 
    previousRepCount: 0, 
    lastDataSnapshot: null, 

    hideAllDots: function() {
        const dots = document.querySelectorAll('.glowing-dot');
        dots.forEach(d => d.classList.remove('dot-active'));
    },

    startSessionTimer: function() {
        if(sessionTimerInterval) clearInterval(sessionTimerInterval);
        sessionSeconds = 0;
        document.getElementById('session-timer-display').style.display = 'inline-block';
        
        sessionTimerInterval = setInterval(() => {
            sessionSeconds++;
            const mins = String(Math.floor(sessionSeconds / 60)).padStart(2, '0');
            const secs = String(sessionSeconds % 60).padStart(2, '0');
            document.getElementById('session-time').innerText = `${mins}:${secs}`;
        }, 1000);
    },

    stopSessionTimer: function() {
        if(sessionTimerInterval) clearInterval(sessionTimerInterval);
        sessionTimerInterval = null;
    },

    updateSessionStatus: function(status) {
        const statusEl = document.getElementById('connection-status');
        if (status === 'START_EXERCISE') {
            statusEl.innerHTML = '<i class="fas fa-play-circle"></i> 운동 진행 중';
            statusEl.className = "badge bg-primary p-2 fs-6 me-3";
            this.startSessionTimer();
        } else if (status === 'END_EXERCISE') {
            statusEl.innerHTML = '<i class="fas fa-check-circle"></i> 운동 종료 (분석 중)';
            statusEl.className = "badge bg-success p-2 fs-6 me-3";
            this.stopSessionTimer();
        }
    },

    resetUI: function() {
        this.currentExercise = ''; 
        this.previousRepCount = 0; 
        this.hideAllDots(); 
        this.stopSessionTimer();
        if(document.getElementById('session-timer-display')) {
            document.getElementById('session-timer-display').style.display = 'none';
        }
        document.getElementById('exercise_selector').selectedIndex = 0;
        document.getElementById('sys_subtitle').innerText = "대기 중... 운동을 선택해주세요.";
        const feedbackEl = document.getElementById('main_feedback');
        feedbackEl.innerText = '"대기 중입니다..."';
        feedbackEl.style.backgroundColor = "#e3f2fd";
        feedbackEl.style.color = "#0d47a1";
        document.getElementById('rep_count_main').innerText = 0;
        
        if (angleChart) {
            angleChart.data.labels = [];
            angleChart.data.datasets[0].data = [];
            angleChart.data.datasets[1].data = [];
            angleChart.update('none');
        }
    },

    startExerciseUI: function(exerciseValue, exerciseName) {
        this.currentExercise = exerciseValue; 
        this.previousRepCount = 0; 
        this.hideAllDots();
        const selector = document.getElementById('exercise_selector');
        if (selector) selector.value = exerciseValue;
        document.getElementById('sys_subtitle').innerText = `${exerciseName} 실 실시간 모니터링 시스템`;

        if (exerciseValue === 'lateral_raise') {
            document.getElementById('chart_title').innerHTML = '<i class="fas fa-chart-line text-info"></i> 실시간 관절 궤적 (사레레)';
            document.getElementById('main_metric_label_left').innerText = "좌측 최고 도달";
            document.getElementById('main_metric_label_right').innerText = "우측 최고 도달";
            document.getElementById('dot_lat_left')?.classList.add('dot-active');
            document.getElementById('dot_lat_right')?.classList.add('dot-active');
        } else if (exerciseValue === 'shoulder_press') {
            document.getElementById('chart_title').innerHTML = '<i class="fas fa-chart-line text-warning"></i> 실시간 관절 궤적 (숄더 프레스)';
            document.getElementById('main_metric_label_left').innerText = "어깨 평균 각도";
            document.getElementById('main_metric_label_right').innerText = "팔꿈치 평균 각도";
            document.getElementById('dot_press_left')?.classList.add('dot-active');
            document.getElementById('dot_press_right')?.classList.add('dot-active');
        } else if (exerciseValue === 'bicep_curl') {
            document.getElementById('chart_title').innerHTML = '<i class="fas fa-chart-line text-success"></i> 실시간 관절 궤적 (바벨 이두컬)';
            document.getElementById('main_metric_label_left').innerText = "평균 팔꿈치 각도";
            document.getElementById('main_metric_label_right').innerText = "평균 위팔(상완) 각도";
            document.getElementById('dot_curl_left')?.classList.add('dot-active');
            document.getElementById('dot_curl_right')?.classList.add('dot-active');
        }

        const feedbackEl = document.getElementById('main_feedback');
        feedbackEl.innerText = "운동이 시작되었습니다. 준비 자세를 취해주세요.";
        feedbackEl.style.backgroundColor = "#e8f5e9"; 
        feedbackEl.style.color = "#2e7d32";
        document.getElementById('rep_count_main').innerText = 0;
    },

    updateDashboard: function(data) {
        if(!data) return;
        this.lastDataSnapshot = data; 
        if(data.system_status) this.updateSessionStatus(data.system_status);

        const exType = data.exercise_type;
        if (exType && exType !== this.currentExercise) {
            let exName = exType === 'lateral_raise' ? "사레레" : exType === 'shoulder_press' ? "숄더 프레스" : "바벨 이두컬";
            this.startExerciseUI(exType, exName);
        }

        if(data.rep_count !== undefined) {
            const newCount = parseInt(data.rep_count);
            document.getElementById('rep_count_main').innerText = newCount;
            if (newCount > this.previousRepCount) this.playRewardAnimation();
            this.previousRepCount = newCount; 
        }

        if(data.last_feedback) document.getElementById('main_feedback').innerText = data.last_feedback;

        if (exType === 'lateral_raise') {
            let metrics = data.elderly_pt_metrics || {};
            if(document.getElementById('main_max_rom_left')) {
                document.getElementById('main_max_rom_left').innerText = Math.round(data.realtime_joints?.left_shoulder || metrics.max_rom_left || 0);
                document.getElementById('main_max_rom_right').innerText = Math.round(data.realtime_joints?.right_shoulder || metrics.max_rom_right || 0);
            }
        } else if (exType === 'shoulder_press') {
            if(document.getElementById('main_max_rom_left')) {
                document.getElementById('main_max_rom_left').innerText = Math.round(data.realtime_joints?.left_shoulder || data.avg_shoulder_angle || 0);
                document.getElementById('main_max_rom_right').innerText = Math.round(data.realtime_joints?.right_shoulder || data.avg_elbow_angle || 0);
            }
        } else if (exType === 'bicep_curl') {
            if(document.getElementById('main_max_rom_left')) {
                document.getElementById('main_max_rom_left').innerText = Math.round(data.realtime_joints?.left_shoulder || data.avg_elbow_angle || 0);
                document.getElementById('main_max_rom_right').innerText = Math.round(data.realtime_joints?.right_shoulder || data.avg_upper_arm_angle || 0);
            }
        }

        if(data.realtime_joints) {
            this.updateRealtimeChart(data.realtime_joints.left_shoulder, data.realtime_joints.right_shoulder);
        }
    },

    playRewardAnimation: function() {
        confetti({ particleCount: 150, spread: 90, origin: { y: 0.6 }, colors: ['#FFD700', '#FFA500', '#1E90FF', '#32CD32'], zIndex: 9999 });
    },

    updateConnectionStatus: function(isOnline) {
        const statusEl = document.getElementById('connection-status');
        if (isOnline) {
            statusEl.innerHTML = '<i class="fas fa-wifi"></i> 실시간 연동 중';
            statusEl.className = "badge bg-success p-2 fs-6 me-3";
        } else {
            statusEl.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 연결 대기 중...';
            statusEl.className = "badge bg-secondary p-2 fs-6 me-3";
            this.stopSessionTimer();
        }
    },

    initChart: function() {
        const ctx = document.getElementById('jointAngleChart').getContext('2d');
        if (angleChart) angleChart.destroy(); 

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
                scales: { y: { min: 0, max: 180, title: { display: true, text: 'Angle (Deg)' } }, x: { display: true } },
                plugins: { legend: { position: 'top' } }, animation: false 
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
    },

    generateGrowthReport: function() {
        const anatomyImg = document.getElementById('report_anatomy_img');
        const anatomyPlaceholder = document.getElementById('report_anatomy_placeholder');
        const reportDots = ['report_dot_lat_left', 'report_dot_lat_right', 'report_dot_press_left', 'report_dot_press_right', 'report_dot_curl_left', 'report_dot_curl_right'];

        if (!this.lastDataSnapshot) {
            document.getElementById('report_total_score').innerText = "0";
            document.getElementById('report_table_body').innerHTML = '<tr><td colspan="4" class="py-3 text-muted">데이터를 분석 중입니다...</td></tr>';
            this.updateRadarChart(0, 0, 0);

            // 🌟 데이터가 없으면 이미지와 점을 숨기고 placeholder를 띄움
            if(anatomyImg) anatomyImg.style.display = 'none';
            if(anatomyPlaceholder) anatomyPlaceholder.style.display = 'block';
            reportDots.forEach(id => { const el = document.getElementById(id); if(el) el.style.display = 'none'; });

            return;
        }

        // 🌟 데이터가 들어왔으므로 placeholder 숨기고 이미지를 띄움
        if(anatomyImg) anatomyImg.style.display = 'inline-block';
        if(anatomyPlaceholder) anatomyPlaceholder.style.display = 'none';
        reportDots.forEach(id => { const el = document.getElementById(id); if(el) el.style.display = 'none'; });

        const data = this.lastDataSnapshot;
        const exType = data.exercise_type || this.currentExercise;
        const scores = data.report_scores || {};
        
        let mobilityScore = scores.mobility_score !== undefined ? scores.mobility_score : 0;
        let stabilityScore = scores.stability_score !== undefined ? scores.stability_score : 50;
        let postureAccuracy = scores.posture_accuracy !== undefined ? scores.posture_accuracy : (data.good_posture_ratio || data.performance_stats?.good_posture_ratio || 0);
        
        document.getElementById('report_date').innerText = new Date().toISOString().split('T')[0];
        document.getElementById('report_total_score').innerText = mobilityScore + stabilityScore;
        document.getElementById('report_ai_comment').innerText = data.ai_comment || "모든 세션이 종료된 후 AI가 맞춤형 종합 처방을 제공합니다.";

        let tableHTML = '';

        if (exType === 'lateral_raise') {
            document.getElementById('report_exercise_title').innerText = "3. 세부 지표 및 타겟 부위 분석 (사레레)";
            if(anatomyImg) anatomyImg.src = 'images/body_outline_shoulder.png';
            const dotL = document.getElementById('report_dot_lat_left'); if(dotL) dotL.style.display = 'block';
            const dotR = document.getElementById('report_dot_lat_right'); if(dotR) dotR.style.display = 'block';
            
            let metrics = data.elderly_pt_metrics || {};
            let pStats = data.performance_stats || {};
            let rAssist = data.robot_assist_parameters || {};
            let warns = data.warning_counts || {};
            
            tableHTML += `
                <tr>
                    <td rowspan="4" class="fw-bold bg-light">관절 가동성<br><small class="text-muted fw-normal">Mobility</small></td>
                    <td class="text-start"><strong>최대 도달 각도 (Max ROM)</strong><br><small class="text-muted"></td>
                    <td class="fw-bold text-primary">좌: ${Math.round(metrics.max_rom_left || 0)}°<br>우: ${Math.round(metrics.max_rom_right || 0)}°</td>
                    <td>80° - 90°</td>
                </tr>
                <tr>
                    <td class="text-start"><strong>순수 능동 가동범위 (Pure AROM)</strong><br><small class="text-muted"></td>
                    <td class="fw-bold text-primary">${rAssist.pure_arom || 0}°</td>
                    <td>${rAssist.target_prom || 0}° (목표)</td>
                </tr>
                <tr>
                    <td class="text-start"><strong>로봇 개입 각도 (Assist Trigger)</strong><br><small class="text-muted"></td>
                    <td class="fw-bold text-warning">${rAssist.assist_trigger_angle || 0}°</td>
                    <td>지연 개입 권장</td>
                </tr>
                <tr>
                    <td class="text-start"><strong>1회 평균 소요 시간 (Pace)</strong><br><small class="text-muted"></td>
                    <td class="fw-bold text-secondary">${metrics.avg_rep_duration_sec || 0}초</td>
                    <td>3~4초</td>
                </tr>
                <tr>
                    <td rowspan="4" class="fw-bold bg-light">자세 안정성<br><small class="text-muted fw-normal">Stability</small></td>
                    <td class="text-start"><strong>Z축 전후 흔들림 (Z-Drift)</strong><br><small class="text-muted"></td>
                    <td class="fw-bold text-danger">${metrics.max_z_depth_drift_mm || 0} mm</td>
                    <td>50mm 미만</td>
                </tr>
                <tr>
                    <td class="text-start"><strong>상체 평균 기울기 (Trunk Angle)</strong><br><small class="text-muted"></td>
                    <td class="fw-bold text-danger">${pStats.avg_trunk_angle || 0}°</td>
                    <td>5° 미만</td>
                </tr>
                <tr>
                    <td class="text-start"><strong>미세 떨림 (Tremor Count)</strong><br><small class="text-muted"></td>
                    <td class="fw-bold text-danger">${metrics.tremor_count || 0}회</td>
                    <td>최소화</td>
                </tr>
                <tr>
                    <td class="text-start"><strong>자세 경고 지표 (Warnings)</strong><br><small class="text-muted"></td>
                    <td class="fw-bold text-danger">반동: ${warns.lean_back_momentum || 0}회<br>불균형: ${warns.arm_balance_issue || 0}회</td>
                    <td>5% 미만</td>
                </tr>
                <tr>
                    <td rowspan="2" class="fw-bold bg-light">운동 추적률<br><small class="text-muted fw-normal">Tracking</small></td>
                    <td class="text-start"><strong>정자세 비율 (Good Posture Ratio)</strong><br><small class="text-muted"></td>
                    <td class="fw-bold text-success">${pStats.good_posture_ratio || 0}%</td>
                    <td>80% 이상</td>
                </tr>
                <tr>
                    <td class="text-start"><strong>추적 세션 데이터 (Session Data)</strong><br><small class="text-muted"></td>
                    <td class="fw-bold text-secondary">${data.frame_count || 0} 프레임<br>${data.session_duration_sec || 0}초</td>
                    <td>-</td>
                </tr>
            `;

        } else if (exType === 'shoulder_press') {
            document.getElementById('report_exercise_title').innerText = "3. 세부 지표 및 타겟 부위 분석 (숄더 프레스)";
            if(anatomyImg) anatomyImg.src = 'images/body_outline_shoulder.png';
            const dotL = document.getElementById('report_dot_press_left'); if(dotL) dotL.style.display = 'block';
            const dotR = document.getElementById('report_dot_press_right'); if(dotR) dotR.style.display = 'block';
            
            let warns = data.warning_counts || {};

            tableHTML += `
                <tr>
                    <td rowspan="2" class="fw-bold bg-light align-middle" style="border-bottom: 2px solid #858796;">관절 가동성<br><small class="text-muted fw-normal">Mobility</small></td>
                    <td class="text-start align-middle"><strong>평균 어깨 굽힘 각도 (Shoulder Flexion)</strong><br><small class="text-muted"></td>
                    <td class="fw-bold text-primary align-middle">${Math.round(data.avg_shoulder_angle || 0)}°</td>
                    <td class="align-middle">145° 이상</td>
                </tr>
                <tr style="border-bottom: 2px solid #858796;">
                    <td class="text-start align-middle"><strong>평균 팔꿈치 폄 각도 (Elbow Extension)</strong><br><small class="text-muted"></td>
                    <td class="fw-bold text-primary align-middle">${Math.round(data.avg_elbow_angle || 0)}°</td>
                    <td class="align-middle">160° 이상</td>
                </tr>
                <tr>
                    <td rowspan="3" class="fw-bold bg-light align-middle" style="border-bottom: 2px solid #858796;">자세 안정성<br><small class="text-muted fw-normal">Stability</small></td>
                    <td class="text-start align-middle"><strong>상체 평균 기울기 (Trunk Angle)</strong><br><small class="text-muted"></td>
                    <td class="fw-bold text-danger align-middle">${data.avg_trunk_angle || 0}°</td>
                    <td class="align-middle">5° 미만</td>
                </tr>
                <tr>리 반동: 1회
양팔 불균형: 2회
                    <td class="text-start align-middle"><strong>하강 범위 이탈 (ROM Warnings)</strong><br><small class="text-muted"></td>
                    <td class="fw-bold text-danger align-middle">과도한 내림: ${warns.too_low || 0}회</td>
                    <td class="align-middle">5% 미만</td>
                </tr>
                <tr style="border-bottom: 2px solid #858796;">
                    <td class="text-start align-middle"><strong>보상 작용 및 불균형 (Compensations)</strong><br><small class="text-muted"></td>
                    <td class="fw-bold text-danger align-middle">양팔 불균형: ${warns.arm_balance_issue || 0}회</td>
                    <td class="align-middle">5% 미만</td>
                </tr>
                <tr style="border-bottom: 2px solid #858796;">
                    <td class="fw-bold bg-light align-middle">운동 정확도<br><small class="text-muted fw-normal">Accuracy</small></td>
                    <td class="text-start align-middle"><strong>정자세 비율 (Good Posture Ratio)</strong><br><small class="text-muted"></td>
                    <td class="fw-bold text-success align-middle">${data.good_posture_ratio || 0}%</td>
                    <td class="align-middle">80% 이상</td>
                </tr>
            `;

        } else if (exType === 'bicep_curl') {
            document.getElementById('report_exercise_title').innerText = "3. 세부 지표 및 타겟 부위 분석 (바벨 이두컬)";
            if(anatomyImg) anatomyImg.src = 'images/body_outline_bicep.png';
            const dotL = document.getElementById('report_dot_curl_left'); if(dotL) dotL.style.display = 'block';
            const dotR = document.getElementById('report_dot_curl_right'); if(dotR) dotR.style.display = 'block';

            let warns = data.warning_counts || {};

            tableHTML += `
                <tr>
                    <td rowspan="2" class="fw-bold bg-light">관절 가동성<br><small class="text-muted fw-normal">Mobility</small></td>
                    <td class="text-start"><strong>평균 팔꿈치 수축 각도 (Elbow Flexion)</strong><br><small class="text-muted"></td>
                    <td class="fw-bold text-primary">${Math.round(data.avg_elbow_angle || 0)}°</td>
                    <td>50° 이하 수축</td>
                </tr>
                <tr>
                    <td class="text-start"><strong>평균 위팔 고정 각도 (Upper Arm Angle)</strong><br><small class="text-muted"></td>
                    <td class="fw-bold text-primary">${Math.round(data.avg_upper_arm_angle || 0)}°</td>
                    <td>10° 미만</td>
                </tr>
                <tr>
                    <td rowspan="3" class="fw-bold bg-light">자세 안정성<br><small class="text-muted fw-normal">Stability</small></td>
                    <td class="text-start"><strong>상체 평균 기울기 (Trunk Angle)</strong><br><small class="text-muted"></td>
                    <td class="fw-bold text-danger">${data.avg_trunk_angle || 0}°</td>
                    <td>5° 미만</td>
                </tr>
                <tr>
                    <td class="text-start"><strong>고립 이탈 경고 (Isolation Break)</strong><br><small class="text-muted"></td>
                    <td class="fw-bold text-danger">${warns.elbows_not_close_to_body || 0}회</td>
                    <td>5% 미만</td>
                </tr>
                <tr>
                    <td class="text-start"><strong>보상 작용 및 불균형 (Compensations)</strong><br><small class="text-muted"></td>
                    <td class="fw-bold text-danger">허리 반동: ${warns.body_not_straight || 0}회<br>양팔 불균형: ${warns.arm_balance_issue || 0}회</td>
                    <td>5% 미만</td>
                </tr>
                <tr>
                    <td rowspan="2" class="fw-bold bg-light">운동 추적률<br><small class="text-muted fw-normal">Tracking</small></td>
                    <td class="text-start"><strong>정자세 비율 (Good Posture Ratio)</strong><br><small class="text-muted"></td>
                    <td class="fw-bold text-success">${data.good_posture_ratio || 0}%</td>
                    <td>80% 이상</td>
                </tr>
            `;
        }

        document.getElementById('report_table_body').innerHTML = tableHTML;
        this.updateRadarChart(mobilityScore * 2, stabilityScore * 2, postureAccuracy);
    },

    updateRadarChart: function(mob100, stab100, posture100) {
        const ctx = document.getElementById('reportRadarChart').getContext('2d');
        if (reportChart) { reportChart.destroy(); } 

        reportChart = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['관절 가동성', '자세 안정성', '정자세 정확도'],
                datasets: [{
                    label: '성취도',
                    data: [mob100, stab100, posture100], 
                    backgroundColor: 'rgba(30, 144, 255, 0.3)',
                    borderColor: 'rgba(30, 144, 255, 1)',
                    pointBackgroundColor: 'rgba(30, 144, 255, 1)',
                    borderWidth: 2
                }]
            },
            options: { scales: { r: { min: 0, max: 100, ticks: { display: false } } }, plugins: { legend: { display: false } } }
        });
    }
};