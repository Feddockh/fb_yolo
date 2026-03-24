# Export YOLOv8 to ONNX (raw outputs; do NMS yourself in C++)
yolo export model=/path/to/best.pt format=onnx imgsz=1088,1440 opset=17 simplify=True dynamic=False nms=False

yolo export model=/home/hayden/cmu/kantor_lab/fb_models/fb_yolo/runs/train/yolov8_large_rivendale_v6_k_fold/yolov8_large_rivendale_v6_k_fold4/weights/best.pt format=onnx imgsz=1088,1440 opset=17 simplify=True dynamic=False nms=False