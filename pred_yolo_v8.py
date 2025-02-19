from ultralytics import YOLO
import os
import cv2
import time

# 시작 시간 기록   ##### 작업 시간 측정 용도
start_time = time.time()






################# 초기 폴더 위치 및 생성 폴더 이름 ##########################
# 모델 경로
weight_path = r"C:\Users\HOME\Desktop\Head-Detection-Yolov8-main\models\weights\best.pt"

# Load a model
model = YOLO(weight_path)  # load a pretrained model (recommended for training)

image_folder = "images"  # 입력 이미지 폴더
label_image_folder = "label_image"  # 라벨링된 이미지 저장 폴더
information_folder = "information"  # 바운딩 박스 정보 저장 폴더


#####################################################################################




################ 모델 박스 크기 설정 및 탐지 기준#######################################

### 최소 박스 탐지 크기 
##### 현재 가로 30px 세로 30px 이상 크기 박스만 탐지 가능


box_width=20
box_height=20

###### 탐지되는 신뢰도 기준   
####0부터 1까지 값

box_conf=0.3

#####################################################################################

# 폴더 생성
os.makedirs(label_image_folder, exist_ok=True)
os.makedirs(information_folder, exist_ok=True)

# 이미지 처리
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)

    # 이미지 읽기
    image = cv2.imread(image_path)

    # 모델 추론
    results = model(image)

    # 결과 파일 경로 설정
    labeled_image_path = os.path.join(label_image_folder, image_name)
    info_file_path = os.path.join(information_folder, f"{os.path.splitext(image_name)[0]}.txt")

    # 바운딩 박스 정보 저장
    with open(info_file_path, "w") as info_file:
        for result in results:
            boxes = result.boxes.xyxy  # 바운딩 박스 좌표 (x1, y1, x2, y2)
            confidences = result.boxes.conf  # 신뢰도 점수

            for box, confidence in zip(boxes, confidences):
                x1, y1, x2, y2 = map(int, box)
                confidence = float(confidence)
                print('confidence=',confidence)
                print('\nwidth=',x2-x1)
                print('height=',y2-y1)

                # 바운딩 박스 크기 필터링 (너비와 높이 특정 크기이상 + 신뢰도 점수 특정 이상인 경우만 처리)
                if (x2 - x1 > box_width) and (y2 - y1 > box_height) and confidence >=box_conf:
                    # 바운딩 박스 정보 파일에 저장
                    
                    info_file.write(f"{x1} {y1} {x2} {y2}\n")

                    # 이미지에 바운딩 박스 그리기
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 라벨링된 이미지 저장
    cv2.imwrite(labeled_image_path, image)
    print(f"Processed and saved: {labeled_image_path}")

# 총 실행 시간 계산 및 출력
end_time = time.time()
execution_time = end_time - start_time
print(f"All images have been processed and labeled.")
print(f"Total execution time: {execution_time:.2f} seconds")
