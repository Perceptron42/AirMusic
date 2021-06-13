import numpy as np
import time
import cv2
from pygame import mixer


# The function compares the total number of detected blue pixels with a threshold and if the sum is larger than
# threshold the corresponding sound is played by the line drum_clap.play() or `drum_snare.play().
def state_machine(sum_white, sound):
    # Check if blue color object present in the ROI
    yes = sum_white > Hatt_thickness[0] * Hatt_thickness[1] * 0.8

    # If present play the respective instrument.
    if yes and sound == 1:
        drum_clap.play()

    elif yes and sound == 2:
        drum_snare.play()
        # This is the number of seconds execution to be suspended.
        time.sleep(0.001)


# Region of Interest Analysis
def ROI_analysis(frame, sound):
    #Frame corresponding ROI is cropped and passed to the function which performs all the operations to
    # detect blue colour pixels in it. We use HSV colour space to detect the presence of blue colour.
    # converting the image into HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # generating mask for pixels corresponding to detected blue colour.
    # Checks if array elements lie between the elements of two other arrays.
    mask = cv2.inRange(hsv, blueLower, blueUpper)

    # Calculating the number of white pixels depicting the blue color pixels in the ROI
    sum_white = np.sum(mask)

    # Function that decides to play the instrument or not.
    state_machine(sum_white, sound)

    return mask

Verbsoe = False #verbose is a Boolean that you can set to visualise the processing in the ROI (Region of interest)

# importing the audio files
mixer.init()
drum_clap = mixer.Sound('batterrm.wav')
drum_snare = mixer.Sound('button-2.ogg')
# With the above lines we set the values to detect the blue colour.
# These values will be used in the [creating mask](# creating mask ) to find pixels corresponding to blue colour inside the ROI.

# HSV range for detecting blue color
# HSV (hue, saturation, value) colorspace is a model to represent the colorspace similar to the RGB color model.
# Since the hue channel models the color type, it is very useful in image processing tasks that need to segment objects
# based on its color. Variation of the saturation goes from unsaturated to represent shades of gray and
# fully saturated (no white component). Value channel describes the brightness or the intensity of the color.
# https://docs.opencv.org/3.4/da/d97/tutorial_threshold_inRange.html
blueLower = (80,150,10)
blueUpper = (120,255,255)

# Frame acquisition from webcam
camera = cv2.VideoCapture(0)
ret,frame = camera.read()
H,W = frame.shape[:2]

print("Original size of your webcam is", frame.shape)

kernel = np.ones((7,7),np.uint8)

# reading the image of hatt and snare for augmentation.
Hatt = cv2.resize(cv2.imread('./images/Hatt.png'),(200,100),interpolation=cv2.INTER_CUBIC)
Snare = cv2.resize(cv2.imread('./images/Snare.png'),(200,100),interpolation=cv2.INTER_CUBIC)
#to read an image the syntax is cv2.imread("path of the image") to reshape the image as per desired shape we use
# cv2.resize(image,(width,heigh),interpolation=INTERPOLATION_METHOD_FLAG_VALUE) The size for augmenting the
# objects is decided based on the ROI.

# Setting the ROI area for blue color detection
Hatt_center = [np.shape(frame)[1]*2//8,np.shape(frame)[0]*6//8]
Snare_center = [np.shape(frame)[1]*6//8,np.shape(frame)[0]*6//8]
Hatt_thickness = [200,100]
Hatt_top = [Hatt_center[0]-Hatt_thickness[0]//2,Hatt_center[1]-Hatt_thickness[1]//2]
Hatt_btm = [Hatt_center[0]+Hatt_thickness[0]//2,Hatt_center[1]+Hatt_thickness[1]//2]

Snare_thickness = [200,100]
Snare_top = [Snare_center[0]-Snare_thickness[0]//2,Snare_center[1]-Snare_thickness[1]//2]
Snare_btm = [Snare_center[0]+Snare_thickness[0]//2,Snare_center[1]+Snare_thickness[1]//2]

#Why ROI is needed?
# Answer is Speed. To detect blue colour we need to perform certain operations on each captured frame.
# These operations need some computations to be performed by the processor. Since our instruments are fixed in this application and we want to play the sound only if the blue colour object hits the instrument (detected inside the ROI) it is a good idea to perform all these operations only inside the ROI.
# With the below lines of code we calculate the top left and bottom right corners of the ROI corresponding to both
# the instruments Hatt and Snare.

# This is the number of seconds execution to be suspended.
time.sleep(1)

while True:

    # capture current camera frame and store it in a numpy array.
    ret, frame = camera.read()
    #Flip the image so the output is a mirror image of yours
    frame = cv2.flip(frame, 1)

    if not (ret):
        break

    # Selecting ROI corresponding to snare
    snare_ROI = np.copy(frame[Snare_top[1]:Snare_btm[1], Snare_top[0]:Snare_btm[0]])
    mask = ROI_analysis(snare_ROI, 1)

    # Selecting ROI corresponding to Hatt
    hatt_ROI = np.copy(frame[Hatt_top[1]:Hatt_btm[1], Hatt_top[0]:Hatt_btm[0]])
    mask = ROI_analysis(hatt_ROI, 2)

    # Selecting the ROI corresponding to Snare and Hatt. ROI is selected by indexing the rows and columns of the image frame.
    # img[a:b, c:d]
    # The above line is an example how you can index rows from a to b of columns c to d of the image stored as a numpy array img.


    # A writing text on an image.
    # https://www.geeksforgeeks.org/python-opencv-cv2-puttext-method/
    cv2.putText(frame, 'Air Music', org = (10, 30), fontFace= 1, fontScale = 2, color = (137,143,139), thickness = 1)

    # Display the ROI to view the blue color being detected
    if Verbsoe:
        # Displaying the ROI in the Image
        frame[Snare_top[1]:Snare_btm[1], Snare_top[0]:Snare_btm[0]] = cv2.bitwise_and(
            frame[Snare_top[1]:Snare_btm[1], Snare_top[0]:Snare_btm[0]],
            frame[Snare_top[1]:Snare_btm[1], Snare_top[0]:Snare_btm[0]],
            mask=mask[Snare_top[1]:Snare_btm[1], Snare_top[0]:Snare_btm[0]])
        frame[Hatt_top[1]:Hatt_btm[1], Hatt_top[0]:Hatt_btm[0]] = cv2.bitwise_and(
            frame[Hatt_top[1]:Hatt_btm[1], Hatt_top[0]:Hatt_btm[0]],
            frame[Hatt_top[1]:Hatt_btm[1], Hatt_top[0]:Hatt_btm[0]],
            mask=mask[Hatt_top[1]:Hatt_btm[1], Hatt_top[0]:Hatt_btm[0]])

    # Augmenting the instruments in the output frame.
    else:
        # Augmenting the image of the instruments on the frame.
        frame[Snare_top[1]:Snare_btm[1], Snare_top[0]:Snare_btm[0]] = cv2.addWeighted(Snare, 1,
                                                                                      frame[Snare_top[1]:Snare_btm[1],
                                                                                      Snare_top[0]:Snare_btm[0]], 1, 0)
        frame[Hatt_top[1]:Hatt_btm[1], Hatt_top[0]:Hatt_btm[0]] = cv2.addWeighted(Hatt, 1,
                                                                                  frame[Hatt_top[1]:Hatt_btm[1],
                                                                                  Hatt_top[0]:Hatt_btm[0]], 1, 0)

    cv2.imshow('Output', frame)
    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()

# code modified from this repo: https://github.com/kaustubh-sadekar/AIR_Drums
# https://github.com/kaustubh-sadekar/AIR_Drums/blob/master/LICENSE
