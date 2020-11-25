import numpy as np
from cv2 import cv2
import math
import bluetooth
import time
from threading import Thread
import queue
#dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)


dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters_create()


cameraMatrix = np.matrix([[1.04435155e+03, 0.00000000e+00, 5.69139364e+02],[0.00000000e+00, 1.04329204e+03, 3.93838599e+02],[  0.00000000e+00, 0.00000000e+00, 1.00000000e+00 ]])
distCoeffs =  np.array([[ -0.11558785,  0.67470449,  0.0093756,  -0.01775784, -1.29717135]])


cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
rvec = np.array([])
tvec = np.array([])
dst = np.array([1000,600])
vec = np.array([])

import SimulatedAnnealing_copy
#g,m = SimulatedAnnealing_copy.SA(plot=True)
#print(m)
#coords=SimulatedAnnealing_copy.SA()
#z1=int(np.dot(coords[0][0],720))
#z2=int(np.dot(coords[0][1],720)) 
#print(z1)
#print(z2)
# print(coords[1])
#import Path_Single_copy
#pa,coords,d = Path_Single_copy.graph_path(m[0],m[1],plot=True)
#print(coords[1][1])

def BT_connect():
    startTime = time.time()
    #Look for all Bluetooth devices the computer knows about.
    print ("Searching for devices...")
    print ("")
    #Create an array with all the MAC addresses of the detected devices
    nearby_devices = bluetooth.discover_devices()

    #Run through all the devices found and list their name
    num = 0
    print ("Select your device by entering its coresponding number...")
    for i in nearby_devices:
        num+=1
        print (num , ": " , bluetooth.lookup_name( i ))

    #Allow the user to select their bluetooth module. They must have paired it before hand.
    selection = int(input("> ")) - 1
    print ("You have selected", bluetooth.lookup_name(nearby_devices[selection]))
    bd_addr = nearby_devices[selection]

    port = 1
    sock = bluetooth.BluetoothSocket( bluetooth.RFCOMM )
    sock.connect((bd_addr, port))
    return(sock)




def angles_from_rvec(rvecs):
    r_mat, _jacobian = cv2.Rodrigues(rvecs)
    a = math.atan2(r_mat[2][1], r_mat[2][2])
    b = math.atan2(-r_mat[2][0], math.sqrt(math.pow(r_mat[2][1],2) + math.pow(r_mat[2][2],2)))
    c = math.atan2(r_mat[1][0], r_mat[0][0])
    return [a,b,c]

def calc_heading(rvecs):
    angles = angles_from_rvec(rvecs)
    degree_angle = math.degrees(angles[2])
    if degree_angle < 0 :
        degree_angle = 360 + degree_angle
    return degree_angle

def aruco_detect(frame):


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters, cameraMatrix=cameraMatrix, distCoeff=distCoeffs)
    #result = set() # create an empty set for result
    result = []
    i=0

    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids,(127,255,0))
        rvec, tvec ,_ = cv2.aruco.estimatePoseSingleMarkers(corners, .15, cameraMatrix, distCoeffs)
        for i in range(0,len(ids)):
            #cv2.aruco.drawAxis(frame, cameraMatrix, distCoeffs,rvec[i],tvec[i],0.1)

            #calc center
            C = np.squeeze(corners[i])

            x1,y1 = C[0] # upper left corner
            x2,y2 = C[2] # lower right corner

            x = int((x1 + x2) / 2)
            y = int((y1 + y2) / 2)

            bearing = calc_heading(rvec[i][0])

            #result.add((ids[i][0], x, y, bearing))
            result = [ids[i][0], x, y, bearing]

            i+=1

            #if [1] in ids:
            #    result = np.where(ids == 1)
            #    print(result)
            #    if i == result[0]:
            #        strPos="Marker1 Pos x=%f y=%f z=%f"%(tvec[i][0][0],tvec[i][0][1],tvec[i][0][2])
            #        cv2.putText(frame, strPos, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2, cv2.LINE_AA )
    return result

def controls(q):
    sock = BT_connect()
    while True:
        phi = q.get()
        print(phi)
        
        if phi < 5:
            data = "A"
            sock.send(data)
            time.sleep(.1)
            sock.send("Z")
            time.sleep(.1)
        elif 5 < phi < 180:
            data = "H"  #left
            sock.send(data)
            time.sleep(.1)
            sock.send("Z")
            time.sleep(.1)
            print(data)
        elif 180 < phi < 355:
            data = "B"#right
            sock.send(data)
            time.sleep(.1)
            sock.send("Z")
            time.sleep(.1)
        elif phi > 355:
            data = "A"#move forward
            sock.send(data)
            time.sleep(.1)
            sock.send("Z")
            time.sleep(.1)
    #    mag = sqrt((vec1**2)+(vec2**2))
    #    if mag < 5:
    #        update dst
        else:
            data = "Z"
            sock.send(data)
            print(data)
#def controls( phi):
#    if phi < 5:
#        data = "Y"
#        sock.send(data)
#    elif 5 < phi < 180:
#        turn left
#    elif 180 < phi < 355:
#        turn right
#    elif phi > 355:
#        move forward
#    mag = sqrt((vec1**2)+(vec2**2))
#    if mag < 5:
#        update dst
#       if phi.full():
#            pass
#        else:
#            data = "Z"
#            sock.send(data)


# Program to find Closest number in a list 
def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)


def calc_dst(q):
    ##
    #import SimulatedAnnealing_copy
    #m = SimulatedAnnealing_copy.SA(plot=True)
    g,m = SimulatedAnnealing_copy.SA(plot=False)
    ##
    
    i=0
    b=0
    cn=0
    c_n=m[-1]
    ##
    import Path_Single_copy
    #pa,coords,d = Path_Single_copy.graph_path(plot=False)
    pa,coords,d = Path_Single_copy.graph_path(m[0],m[1],plot=False)
    ##
    #z1=int(np.dot(coords[0][0],720))
    #z2=int(np.dot(coords[0][1],720))
    while True:
        ##
        if i == (len(coords)):
            #import Path_Single_copy
            #pa,coords,d = Path_Single_copy.graph_path(plot=False)
            b+=1
            
            if b == (len(m)-1):
                c_n=g[cn]
                g,m = SimulatedAnnealing_copy.SA(plot=False)
                b=0
                m.insert(0,c_n)
            
            pa,coords,d = Path_Single_copy.graph_path(m[b],m[b+1],plot=False)
            i=0
        ##
        #z1=int(np.dot(coords[i][0],720))
        #z2=int(np.dot(coords[i][1],720))
        z1=int(coords[i][0])
        z2=int(coords[i][1])
        #z1=int(np.dot(coords[i][0],0.15)+10)
        #z2=int(np.dot(coords[i][1],0.15)+10)
        ret, frame = cap.read()
        val = aruco_detect(frame)
        if val is not None and len(val) > 0:
            #vec1, vec2 = (dst[0]-val[1], dst[1]-val[2])
            vec1, vec2 = (z1-val[1], z2-val[2])
            #print(vec1, vec2)
            theta = math.atan2(vec2,vec1)*180/3.14 # angle between vector(marker->dst) and x-axis
            if theta < 0 :
                theta = 360 + theta
            #print(theta)
            #cv2.line(frame,(val[1],val[2]), (1200, val[2]), (0,255,100), 2)
            #cv2.line(frame,(val[1],val[2]), (1000, 600), (0,255,0), 2)
            cv2.line(frame,(val[1],val[2]), (1200, val[2]), (0,255,100), 2)
            cv2.line(frame,(val[1],val[2]), (z1, z2), (0,255,0), 2)
            phi = (val[3]-90)-theta # angle between marker heading and vector(marker->dst)
            if phi < 0 :
                phi = 360 + phi
            print(phi)
            if q.full():
                pass
            else:
                q.put(phi)
            mag = np.sqrt((vec1**2)+(vec2**2)) # magnitude
            #if q.empty() and mag < 50: 
            if mag < 25: # if magnitude is less than 50
                #update_dst()
                i+=1 #iterate through list of coordinates
                #break 
            # elif q.empty() and mag > 5
            #   q.put(phi)
            # else: 
            #   pass
            ## find closest node and print on to screen
            cn=closest_node((val[1],val[2]),g)
            cv2.putText(frame, "closest node {}".format(cn), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3) #font stroke
        #print(val)
        #i+=1
        
        #cv2.circle(frame, (dst[0], dst[1]), 5, (255,0,0), -1)
        cv2.circle(frame, (z1, z2), 5, (255,0,255), -1)
        ## draw nodes on path
        for j in range(len(coords)):
            p1=int(coords[j][0])
            p2=int(coords[j][1])
            #p1=int(np.dot(coords[j][0],0.15)+10)
            #p2=int(np.dot(coords[j][1],0.15)+10)
            cv2.circle(frame, (p1, p2), 5, (255,0,255), -1)
        ##
        #cv2.circle(frame, (1000, 600), 5, (255,0,0), -1)
        ##
        ## draw nodes on map
        for k in range(len(g)):
            g1=int(g[k][0])
            g2=int(g[k][1])
            #g1=int(np.dot(g[k][0],0.15)+10)
            #g2=int(np.dot(g[k][1],0.15)+10)
            cv2.circle(frame, (g1, g2), 5, (0,0,255), 1)
        ##

        cv2.imshow('frame', frame)
        #key = cv2.waitKey(1)
        #if key==27:
        if cv2.waitKey(1) == 27:
            #sock.send("Z")
            break
        #return phi




q=queue.Queue(maxsize=1)
phi=0
t1=Thread(target=calc_dst, args=(q,))
#t2=Thread(target=controls, args=(q,))


t1.start()
#t2.start()

t1.join()
#t2.join()


cap.release()
cv2.destroyAllWindows()
#sock.close()
