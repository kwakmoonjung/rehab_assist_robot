import json
import os
import cv2
import numpy as np

def main():
    json_path = "data/calibrate_data.json"
    
    if not os.path.exists(json_path):
        print(f"에러: {json_path} 파일을 찾을 수 없습니다.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    print("=" * 50)
    print("[사용 방법]")
    print("1. 마우스로 체커보드가 있는 영역을 넉넉하게 드래그합니다.")
    print("2. 스페이스바(Space) 또는 엔터(Enter)를 누르면 배경이 하얗게 지워집니다.")
    print("3. 결과를 확인한 뒤, 👉 '오른쪽 방향키'를 누르면 다음 사진으로 넘어갑니다.")
    print("   (방향키 인식이 안 될 경우 영어 'n' 키를 누르세요)")
    print("4. 스킵하고 싶다면 드래그 없이 엔터를 치고 오른쪽 방향키를 누르세요.")
    print("5. 강제 종료하고 싶다면 'q' 또는 'ESC'를 누르세요.")
    print("=" * 50)

    success_count = 0
    for file_name in data["file_name"]:
        img_path = os.path.join("data", file_name)
        
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        window_name = f"Image: {file_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # 1. 사용자가 박스를 그리고 엔터를 칠 때까지 대기
        roi = cv2.selectROI(window_name, img, showCrosshair=True, fromCenter=False)
        x, y, w, h = roi
        
        if w == 0 or h == 0:
            print(f"스킵됨: {file_name} (다음으로 가려면 오른쪽 방향키 입력)")
            result_img = img  # 원본 유지
        else:
            # 2. 선택된 영역 외의 배경을 하얗게 제거
            result_img = np.ones_like(img) * 255
            result_img[y:y+h, x:x+w] = img[y:y+h, x:x+w]
            
            # 덮어쓰기 저장
            cv2.imwrite(img_path, result_img)
            print(f"배경 제거 완료: {file_name}")
            success_count += 1

        # 3. 배경이 지워진 결과 이미지를 화면에 다시 띄워서 보여줌
        cv2.imshow(window_name, result_img)
        
        # 4. 오른쪽 방향키를 누를 때까지 무한 대기
        while True:
            # 방향키 입력을 받기 위해 확장 키 입력 대기 함수 사용
            key = cv2.waitKeyEx(0)
            
            # 'q' (113) 또는 ESC (27) 누르면 전체 작업 강제 종료
            if key == ord('q') or key == 27:
                print("\n작업을 강제 종료합니다.")
                cv2.destroyAllWindows()
                return
                
            # 오른쪽 방향키 (리눅스: 65363, 윈도우: 2555904) 또는 'n' (110)
            if key in [65363, 2555904, ord('n')]:
                break

        # 다음 사진으로 넘어가기 위해 현재 창 닫기
        cv2.destroyWindow(window_name)
        cv2.waitKey(1) # 우분투 GUI 에러 방지용 숨통 트기

    cv2.destroyAllWindows()
    print("=" * 50)
    print(f"작업 완료! 총 {success_count}장의 이미지 배경을 제거했습니다.")

if __name__ == "__main__":
    main()