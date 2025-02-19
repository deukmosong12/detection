# Head-Detection-Yolov8
욜로V8을 학습시켜 나오는 weight들로 탐지 후 바운딩 박스 값들을 저장하는 코드

# Detection Example
![image](https://github.com/user-attachments/assets/074da0a4-343f-4344-a0a2-88d67c23dae9)



# Pre-trained YoloV8 Head Detection Model
Please download the model weight from this [Google Drive URL](https://drive.google.com/file/d/1qlBmiEU4GBV13fxPhLZqjhjBbREvs8-m/view?usp=sharing). 


# 모델 라벨링 코드
`
python3 label.py
`


# annotaiton 화 하는 코드
`
python3 anno.py
`


# Some Useful Scrip as References
`Scrip/create_chuman.py` -- Creat txt format labels from ODGT.
`Scrip/gen_labels.py` -- Generate YOLO labels from txt format.
`Scrip/vis_labels.py`-- Visualize YOLO labels for label checking.

