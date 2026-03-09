// Firebase 프로젝트 환경 설정 객체
// TODO: 본인의 rehab-aa1ee 프로젝트 설정값으로 변경하세요!
const firebaseConfig = { 
    apiKey: "AIzaSyDNUDW9qkE893MCDD1wma8b8yYHLuDW6JQ", 
    authDomain: "rehab-aa1ee.firebaseapp.com", 
    databaseURL: "https://rehab-aa1ee-default-rtdb.firebaseio.com", 
    projectId: "rehab-aa1ee", 
    storageBucket: "rehab-aa1ee.firebasestorage.app", 
    messagingSenderId: "199519896475", 
    appId: "1:199519896475:web:a884328018382df721d73e" 
};

// 앱 초기화 보장
if (!firebase.apps.length) {
    firebase.initializeApp(firebaseConfig);
}

// 데이터베이스 객체 생성
const database = firebase.database();