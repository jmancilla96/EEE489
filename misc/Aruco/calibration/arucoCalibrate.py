import sys
from cv2 import cv2
import numpy as np

#dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
#dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
#board = cv2.aruco.CharucoBoard_create(6,8,.04,.02,dictionary)
board = cv2.aruco.CharucoBoard_create(3,3,.053,.0265,dictionary)
img = board.draw((600,800))

#Dump the calibration board to a file
cv2.imwrite('charuco.png',img)

cap = cv2.VideoCapture(0)


allCharucoCorners = []
allCharucoIds = []

i=0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    # parameters = cv2.aruco.DetectorParameters_create()

    # take pictures and append detected corners and ids to array
    if cv2.waitKey(1) == 32:    #if 'space' is pressed
        i+=1
        cv2.imwrite('aruco/cal/cal{}.png'.format(i), gray)
        print("image {} saved for calibration".format(i))

        im=cv2.imread('aruco/cal/cal{}.png'.format(i))
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(im, aruco_dict)
        if ids is not None and len(ids) > 0:
            retval, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        
            if charucoCorners is not None and charucoIds is not None:
                allCharucoCorners.append(charucoCorners)
                allCharucoIds.append(charucoIds)
        
    # for visual
    viewcorners, viewids, viewrejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)
    if viewids is not None and len(viewids) > 0:
        cv2.aruco.drawDetectedMarkers(gray, viewcorners, viewids)
    cv2.imshow('frame', gray)

    #key = cv2.waitKey(1)
    #if key==27:
    if cv2.waitKey(1) == 27:
        break
   # elif key==32:
   #     i+=1
   #     cv2.imwrite('C:/Users/Javier/Documents/opencvtest/aruco/cal/cal{}.png'.format(i), frame)
   #     print("image {} saved for calibration".format(i))
    





imsize=gray.shape
print(imsize)


retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(allCharucoCorners, allCharucoIds, board, imsize, None, None)


cap.release()
cv2.destroyAllWindows()

f = open("cameraCalibration.txt", "a")
print("cmx: {}".format(cameraMatrix), file=f)
print("distCoeffs: {}".format(distCoeffs), file=f)
print("rvecs: {}".format(rvecs), file=f)
print("tvecs: {}".format(tvecs), file=f)
f.close()