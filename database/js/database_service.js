// Firebase DB 통신 리스너 설정

// 🌟 [수정됨] 종목에 상관없이 현재 진행 중인 '라이브 세션' 경로만 감시합니다.
// (목록이 아니라 단일 객체이므로 orderByKey나 limitToLast가 필요 없습니다)
const sessionsRef = database.ref('live_current_session');

// 🌟 여기서부터가 Firebase DB의 변화를 감지하는 '리스너' 부분입니다 🌟
sessionsRef.on('value', (snapshot) => {
    const data = snapshot.val();

    // 🌟 1. 데이터가 존재할 때 (로봇/파이썬이 데이터를 쏘고 있을 때)
    if (data) {
        // 데이터가 들어오면 '연결됨' 상태로 UI 변경
        UIManager.updateConnectionStatus(true);

        // 통합 파싱 함수(updateDashboard)로 데이터를 통째로 넘깁니다.
        UIManager.updateDashboard(data);
    } 
    // 🌟 2. 데이터가 비어있을 때 (터미널에서 Ctrl+C로 종료하여 DB가 지워졌을 때)
    else {
        // 연결 끊김 상태로 변경하고 화면을 초기화(0) 합니다.
        UIManager.updateConnectionStatus(false);
        UIManager.resetUI();
    }
});