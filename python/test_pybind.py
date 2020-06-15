import os
import sys

lib_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.join(lib_dir, '..', "build")
print(lib_dir)
sys.path.insert(0, lib_dir)
import pytest
from py_darknet import Tkdnn_darknet
import cv2

def test_add():
    img = cv2.imread("dog.jpg", cv2.IMREAD_COLOR)
    yolo = Tkdnn_darknet("/home/msis_dev/Project/tkDNN/build/yolo4_fp16.rt", 80, 1)
    yolo.queue_image(img)
    res = yolo.inference(0)
    print(res)

if __name__ == "__main__":
    test_add()
