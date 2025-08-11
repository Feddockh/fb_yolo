from roboflow import Roboflow

rf = Roboflow(api_key="MIU1JTrOBgTWPdRvn75R")
project = rf.workspace("hfeddock").project("yolov8-fireblight")

# Grab the version object
version = project.version(3)

# Deploy
version.deploy("yolov8", "./runs/train/yolov8_fireblight_large_4")
