import cv2
import numpy
import matplotlib.pyplot as plt
import mediapipe as mp
import os

faceDetect = mp.solutions.face_detection
detection = faceDetect.FaceDetection()
RESIZE_DIM = 34

def detect_face_boundaries(img):
    IMG_H, IMG_W = img.shape[0], img.shape[1]
    face_detected = False
    boundary_tuple = None
    result = detection.process(img)
    
    if result.detections:
        face = result.detections[0]
        box_info = face.location_data.relative_bounding_box
        x,y,w,h = int(box_info.xmin*IMG_W), int(box_info.ymin*IMG_H), int(box_info.width*IMG_W), int(box_info.height*IMG_H)
        boundary_tuple = (y,y+h,x,x+w)
        face_detected = True
        
    return boundary_tuple, face_detected

def apply_jpg_transform(img, color=False):
    # detect face and crop
    face_boundaries, face_detected = detect_face_boundaries(img)
    if not face_detected:
        return None, False
    
    R_START, R_END, C_START, C_END = face_boundaries
    # apply contrast balancing
    img = automatic_brightness_and_contrast(img)
    
    if not color:
        # Grayscale Processing
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    elif color:
        # Color Processing
        pass
    
    cropped_face = img[R_START:R_END, C_START:C_END]
    
    downsampled = cv2.resize(cropped_face, (RESIZE_DIM, RESIZE_DIM), interpolation = cv2.INTER_CUBIC)
    # get the width of the photo
    n = downsampled.shape[1] 
    n = n-1 if n%2==1 else n
    downsampled = cv2.resize(downsampled,(n, n))
    return downsampled, True


def automatic_brightness_and_contrast(image, clip_hist_percent=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)
    
    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))
    
    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0
    
    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    
    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    
    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    
    '''
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    '''

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result