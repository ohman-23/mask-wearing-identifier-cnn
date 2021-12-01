import cv2
import mediapipe as mp
import image_preprocessing_utils as utils
import numpy as np

GRAY_MASK_DIR = "./video_capture_gray_mask"
GRAY_NO_MASK_DIR = "./video_capture_gray_no_mask"
COLOR_MASK_DIR = "./video_capture_color_mask"
COLOR_NO_MASK_DIR = "./video_capture_color_no_mask"

faceDetect = mp.solutions.face_detection
detection = faceDetect.FaceDetection()

UID = int(np.random.uniform(0, 100000))
COUNTER = 0

cap = cv2.VideoCapture(0)


while True:
    _, img = cap.read()

    # get detected faces in a frame for color and grayscale
    face_color, processed_color = utils.apply_jpg_transform(img, color=True)
    face_gray, processed_gray = utils.apply_jpg_transform(img, color=False)

    bounds, face_detected = utils.detect_face_boundaries(img)
    if face_detected:
        display_img = img.copy()
        R_START, R_END, C_START, C_END = bounds
        cv2.rectangle(display_img, (C_START, R_START), (C_END, R_END), (255, 0, 0), 2)
        cv2.imshow('display', display_img)


    KEY = cv2.waitKey(10)

    if processed_color and KEY:
        if KEY == ord('1'):
            cv2.imwrite(f'{COLOR_MASK_DIR}/IMG_{UID}{COUNTER}.jpg', face_color)
        elif KEY == ord('0'):
            cv2.imwrite(f'{COLOR_NO_MASK_DIR}/IMG_{UID}{COUNTER}.jpg', face_color)
        COUNTER += 1
    
    if processed_gray and KEY:
        if KEY == ord('1'):
            cv2.imwrite(f'{GRAY_MASK_DIR}/IMG_{UID}{COUNTER}.jpg', face_gray)
        elif KEY == ord('0'):
            cv2.imwrite(f'{GRAY_NO_MASK_DIR}/IMG_{UID}{COUNTER}.jpg', face_gray)
        COUNTER += 1

    if KEY == ord('q'):
        break
cap.release()