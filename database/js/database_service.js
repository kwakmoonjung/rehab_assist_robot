// Firebase DB 통신 리스너 설정

// 'lateral_raise_sessions' 경로에서 가장 최근에 업데이트된 세션 1개만 계속 감시합니다.
const sessionsRef = database.ref('lateral_raise_sessions').orderByKey().limitToLast(1);

// 🌟 여기서부터가 Firebase DB의 변화를 감지하는 '리스너' 부분입니다 🌟
sessionsRef.on('value', (snapshot) => {
    // 데이터가 들어오면 '연결됨' 상태로 UI 변경
    UIManager.updateConnectionStatus(true);

    snapshot.forEach((childSnapshot) => {
        const data = childSnapshot.val();

        // [업데이트됨] 기존의 중복되거나 개별적인 UI 업데이트 코드들을 모두 지우고,
        // 새로 만든 통합 파싱 함수(updateDashboard)로 데이터를 통째로 넘깁니다.
        UIManager.updateDashboard(data);
    });
});