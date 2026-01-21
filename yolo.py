from ultralytics import YOLO

# 1. 사전 학습된 YOLOv8 모델 로드
# 'yolov8n.pt'는 가장 작고 빠른 모델(nano-sized)입니다.
model = YOLO('yolov8n.pt')

# 2. 이미지 파일에 대해 객체 인식 수행
# source에 이미지 경로를 지정하고, save=True로 설정하면
# 인식 결과가 results 폴더에 이미지 파일로 저장됩니다.
results = model.predict(source='com.jpg', save=True, conf=0.5)

# 3. 인식 결과 출력
# 검출된 각 객체에 대한 정보를 출력합니다.
for result in results:
    boxes = result.boxes  # Bounding Box 정보 접근
    print(f"이미지 크기: {result.orig_shape}")
    print(f"검출된 객체 수: {len(boxes)}")
    for box in boxes:
        cls_id = int(box.cls[0])  # 클래스 ID
        conf = float(box.conf[0])  # 신뢰도(Confidence)
        class_name = model.names[cls_id]  # 클래스 이름
        print(f"클래스: {class_name}, 신뢰도: {conf:.2f}")
