from roboflow import Roboflow

rf = Roboflow(api_key="sI3nm65XP18o9mxdCrDo")
project = rf.workspace("fire-extinguisher").project("fireextinguisher-z5atr")
version = project.version(2)
dataset = version.download("yolov11")