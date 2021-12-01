import cv2
import mediapipe as mp
from cnn_classes import CustomCNNColor, CustomCNNGrayScale, Utils
import image_preprocessing_utils
import pickle

CNN_GRAYSCALE = 'CNN_GRAY_CUSTOM_V2'
CNN_COLOR = 'CNN_COLOR_CUSTOM_V2'

with open(CNN_GRAYSCALE, 'rb') as file:
    gray_model = pickle.load(file)

with open(CNN_COLOR, 'rb') as file:
    color_model = pickle.load(file)

faceDetect = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
detection = faceDetect.FaceDetection()

# import preprocessing utils []
def preprocess_img(img):
    return image_preprocessing_utils.automatic_brightness_and_contrast(img)

def preprocess_face(img):
    img = cv2.resize(img, (34,34), interpolation=cv2.INTER_CUBIC)
    return img/255

# To capture video from webcam. 
cap = cv2.VideoCapture(0)

while True:
    # Read the frame
    _, img = cap.read()
    # preprocess entire image
    img = preprocess_img(img)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    IMG_H, IMG_W, IMG_C = img.shape
    results = detection.process(img)
    if results.detections:
        # continue with analysis
        face = results.detections[0]
        info = face.location_data.relative_bounding_box
        x, y, w, h = int(info.xmin*IMG_W), int(info.ymin*IMG_H), int(info.width*IMG_W), int(info.height*IMG_H)
        cropped_face = img[y:y+h, x:x+w]
        cropped_face_gray =  gray[y:y+h, x:x+w]

        # preprocess images to fit inputs to CNN
        cropped_face = preprocess_face(cropped_face)
        cropped_face_gray = preprocess_face(cropped_face_gray)

        pred_gray = gray_model.predict(cropped_face_gray)
        pred_color = color_model.predict(cropped_face)
        
        # set how you want to weight the predictions
        pred = (pred_gray+pred_color)/2

        if pred > 0.5:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        else:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, f'Wearing Mask: {pred*100}%', (x,y), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)

    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
# Release the VideoCapture object
cap.release()