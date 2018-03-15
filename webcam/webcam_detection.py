import argparse
import sys, os
sys.path.append("../")

import cv2
import numpy as np

import face_detection_utilities as fdu

import feature_utility as fu

import model.myVGG as vgg

windowsName = 'Screen Show'

parser = argparse.ArgumentParser(description='A live emotion recognition from webcam')
parser.add_argument('--testImage', help=('Given the path of testing image, the program will predict the result of the image.'
"This function is used to test if the model works well."))
parser.add_argument('--dataset', help=('Input a directory to test model prediction'))

args = parser.parse_args()
FACE_SHAPE = (48, 48)

model = vgg.VGG_16('my_model_weights_83.h5')

emo     = ['Angry', 'Fear', 'Happy',
           'Sad', 'Surprise', 'Neutral']

def refreshFrame(frame, faceCoordinates, emotion):
    if faceCoordinates is not None:
        fdu.drawFace(frame, faceCoordinates)
        fdu.drawText(frame, faceCoordinates, emotion)
    cv2.imshow(windowsName, frame)


def showScreenAndDectect(capture):
    while (True):
        '''
        Capture frame-by-frame
        read() returns two parameters:
            1.The actual video frame read (one frame on each loop)
            2.A return code (The return code tells us if we have run out of frames, which will happen if we are reading from a file. This doesn’t matter when reading from the webcam, since we can record forever, so we will ignore it.)
        '''
        
        flag, frame = capture.read()
        faceCoordinates = fdu.getFaceCoordinates(frame)
        # refreshFrame(frame, faceCoordinates)
        
        
        # Another solution, get multiple faces to show here.
        '''
        flag, frame = capture.read()
        faceCoordinates = fdu.getFaceCoordinates2(frame)
        refreshFrame2(frame, faceCoordinates)
        '''

        if faceCoordinates is not None:
            face_img = fdu.preprocess(frame, faceCoordinates, face_shape=FACE_SHAPE)
            #cv2.imshow(windowsName, face_img)

            input_img = np.expand_dims(face_img, axis=0)
            input_img = np.expand_dims(input_img, axis=0)

            result = model.predict(input_img)[0]
            index = np.argmax(result)
            print (emo[index], 'probability is: ', max(result))

            refreshFrame(frame, faceCoordinates, emo[index])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    capture.release()
    cv2.destroyAllWindows()

def getCameraStreaming():
    capture = cv2.VideoCapture(0)
    if not capture:
        print("Failed to capture video streaming ")
        sys.exit(1)
    else:
        print("Successed to capture video streaming")

    return capture

def test_single_image(testImage):
    print ('Image Prediction Mode')
    img = fu.preprocessing(cv2.imread(testImage))
    X = np.expand_dims(img, axis=0)
    X = np.expand_dims(X, axis=0)
    result = model.predict(X)[0]
    # print (result)
    index = np.argmax(result)
    print (emo[index], 'probability is: ', max(result))

def test_dataset_image(dataSet):
    print ("Directory Prediction Mode")
    X, y = fu.extract_features(dataSet)
    scores = model.evaluate(X, y, verbose=0)
    print (scores)
    return

def main():
    '''
    Arguments to be set:
        showCam : determine if show the camera preview screen.
    '''
    print("Enter main() function")
    
    # Testing single image.
    if args.testImage is not None:
        test_single_image(args.testImage)
        sys.exit(0)
    elif args.dataset is not None:
        test_dataset_image(args.dataset)
        sys.exit(0)
    showCam = 1

    capture = getCameraStreaming()

    if showCam:
        cv2.startWindowThread()
        cv2.namedWindow(windowsName, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(windowsName, cv2.WND_PROP_FULLSCREEN, cv2.WND_PROP_FULLSCREEN)
    
    showScreenAndDectect(capture)

if __name__ == '__main__':
    main()
