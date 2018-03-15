import cv2
import numpy as np

CASCADE_PATH = "haarcascade_frontalface_default.xml"

RESIZE_SCALE = 3
REC_COLOR = (0, 255, 0)

def getFaceCoordinates(image):

    # Creating a face cascade.
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    
    faces = cascade.detectMultiScale(
        image,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(48, 48)
        )

    if(len(faces) == 0) :
        return None
    # For now, we only deal with the case that we detect one face.
    if(len(faces) != 1) :
        return None
    
    face = faces[0]
    bounding_box = [face[0], face[1], face[0] + face[2], face[1] + face[3]]

    # Suppose we need to detect mulitple faces here, we set the following code.
    '''
    # draw rectangle around the faces.
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    '''

    # return map((lambda x: x), bounding_box)
    return bounding_box

def getFaceCoordinates2(image):

    # Creating a face cascade.
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.equalizeHist(img_gray)
    faces = cascade.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        # Try to set minNeighbors=5 maybe
        minNeighbors=3,
        minSize=(48, 48)
        )
    return faces

def drawFace(img, faceCoordinates):
    drawBorder(img, (faceCoordinates[0], faceCoordinates[1]), \
    (faceCoordinates[2], faceCoordinates[3]), REC_COLOR, 2, 10, 20)

def drawBorder(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2

    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

def drawText(img, faceCoordinates, text):
    cv2.putText(img, text, (faceCoordinates[0], faceCoordinates[1]), cv2.FONT_HERSHEY_PLAIN, 3, REC_COLOR, 2)

def drawFace2(img, faceCoordinates):
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), REC_COLOR, thickness=2)

def crop_face(img, faceCoordinates):
    return img[faceCoordinates[1]:faceCoordinates[3], faceCoordinates[0]:faceCoordinates[2]]

def preprocess(img, faceCoordinates, face_shape=(48, 48)):
    '''
        This function will crop user's face from the original frame
    '''
    face = crop_face(img, faceCoordinates)
    face_scaled = cv2.resize(face, face_shape)
    face_gray = cv2.cvtColor(face_scaled, cv2.COLOR_BGR2GRAY)
    
    return face_gray
