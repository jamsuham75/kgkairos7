from ultralytics import YOLO

# 모델 로드
model = YOLO('yolov8n.pt')

# 신뢰도 임계값을 0.75로 설정하여 예측
# 75% 이상의 신뢰도를 가진 객체만 검출됩니다.
model.predict(source='com.jpg', conf=0.80, save=True)
