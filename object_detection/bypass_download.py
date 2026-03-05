import os
import urllib.request

# 연구원님의 환경에 맞춘 설정
url = "https://huggingface.co/mmpose/rtmpose-m/resolve/main/rtmpose-m_simcc-coco_256x192-f0103637_20230504.onnx"
dest_path = '/home/rokey/cobot_ws/src/cobot2_ws/rehab_assist_robot/object_detection/rtmpose-m.onnx'

def download_model():
    print("🚀 [최종 단계] 라이브러리 우회, 서버 다이렉트 연결 시작...")
    
    # 봇 차단을 막기 위해 브라우저인 척 위장합니다.
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebkit/537.36')]
    urllib.request.install_opener(opener)

    try:
        if os.path.exists(dest_path):
            os.remove(dest_path) # 기존 0바이트 파일 삭제
            
        print(f"📡 모델 다운로드 중: {url}")
        urllib.request.urlretrieve(url, dest_path)
        
        size = os.path.getsize(dest_path) / 1024 / 1024
        print(f"✅ 구출 성공! 파일 위치: {dest_path}")
        print(f"📏 파일 크기: {size:.2f} MB (약 60MB 내외면 정상)")
        
    except Exception as e:
        print(f"❌ 치명적 에러 발생: {e}")
        print("💡 팁: 네트워크 연결을 확인하거나, 브라우저에서 직접 링크를 클릭해 다운로드 후 해당 폴더에 넣으세요.")

if __name__ == "__main__":
    download_model()
