# import os, hashlib, urllib.request, shutil, sys

# ROOT = os.path.dirname(os.path.dirname(__file__))
# MDIR = os.path.join(ROOT, "models")
# os.makedirs(MDIR, exist_ok=True)

# def dl(url, dst, sha256=None):
#     print(f"-> {url}\n   -> {dst}")
#     with urllib.request.urlopen(url) as r, open(dst + ".part", "wb") as f:
#         shutil.copyfileobj(r, f)
#     os.replace(dst + ".part", dst)
#     if sha256:
#         h = hashlib.sha256(open(dst, "rb").read()).hexdigest()
#         if h.lower() != sha256.lower():
#             raise SystemExit(f"SHA256 mismatch for {dst}: {h} != {sha256}")
#         print(f"   sha256 OK: {h}")

# # YuNet (OpenCV)
# YUNET_URL = "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx?download=true"
# YUNET_DST = os.path.join(MDIR, "face_detection_yunet_2023mar.onnx")
# YUNET_SHA = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"  # published
# if not os.path.exists(YUNET_DST):
#     dl(YUNET_URL, YUNET_DST, YUNET_SHA)

# # ArcFace (OpenVINO OMZ)
# ARC_URL = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/face-recognition-resnet100-arcface-onnx/face-recognition-resnet100-arcface-onnx.onnx"
# ARC_DST = os.path.join(MDIR, "arcface.onnx")
# if not os.path.exists(ARC_DST):
#     dl(ARC_URL, ARC_DST, None)  # у них нет опубликованного sha256 рядом

# print("Done.")

from ultralytics import YOLO
m = YOLO("hand_yolov8n.pt")
m.export(format="onnx", opset=12, imgsz=640, simplify=True, dynamic=False)
