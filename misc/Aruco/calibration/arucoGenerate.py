from cv2 import cv2
import numpy as np

#load predefined dictionary
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

#Generate the marker
#markerImage = np.zeros((200,200), dtype=np.uint8)

for i in range(10):
    markerImage = np.zeros((600,600), dtype=np.uint8)
    markerImage = cv2.aruco.drawMarker(dictionary, i, 600, markerImage, 1)
    cv2.imwrite('aruco/markers/marker{}.png'.format(i), markerImage)