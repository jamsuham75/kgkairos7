from ultralytics import YOLO

# 모델 로드
model = YOLO('yolov8n.pt')

# 'images_folder' 폴더 내 모든 이미지 파일에 대해 예측
# 예: 'images_folder' 폴더에 있는 bus.jpg, car.jpg, person.jpg 등
model.predict(source='test', save=True)
