import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tensorflow as tf
import winsound
from eye_cooridinates import *
from prediction import *

def increase_contrast(gray_image):
    hist, bins = np.histogram(gray_image.flatten(), 256, [0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    min_intensity = bins[np.min(np.where(cdf_normalized > 0))]
    max_intensity = bins[np.max(np.where(cdf_normalized < cdf_normalized.max()))]
    desired_min = 100
    desired_max = 160
    adjusted_image = np.interp(gray_image, [min_intensity, max_intensity], [desired_min, desired_max])
    return adjusted_image

base_options = python.BaseOptions(model_asset_path="blaze_face_short_range.tflite")
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

cap = cv2.VideoCapture(0)
t=0
while 1:
    ret,frame =cap.read() 
    rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = detector.detect(rgb_frame)
    result=get_eyes_coordinates(frame,detection_result)
    if result!=None:
        left_eye,right_eye,frame_new=result[0],result[1],result[2]
        left_eye=increase_contrast(left_eye)
        right_eye=increase_contrast(right_eye)
        cv2.imshow('Eyes_detected',frame_new)
        left_eye=np.reshape(left_eye,(1,60,60))
        right_eye=np.reshape(right_eye,(1,60,60))
        if not predict(left_eye,right_eye):
            t+=1
        else:
            t=0
        if t>=3:
            winsound.Beep(2000, 250)

    if cv2.waitKey(250)==27:
        break
cap.release()
cv2.destroyAllWindows()
