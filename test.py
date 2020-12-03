import numpy as np
from cv2 import cv2
import math
import bluetooth
import time
from threading import Thread
import queue
import SimulatedAnnealing_copy
#dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)

# set The ARuco maarker dictionary
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters_create()


cameraMatrix = np.matrix([[1.04435155e+03, 0.00000000e+00, 5.69139364e+02],[0.00000000e+00, 1.04329204e+03, 3.93838599e+02],[  0.00000000e+00, 0.00000000e+00, 1.00000000e+00 ]])
distCoeffs =  np.array([[ -0.11558785,  0.67470449,  0.0093756,  -0.01775784, -1.29717135]])

# Setup camera
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

rvec = np.array([])
tvec = np.array([])
vec = np.array([])


# Below VideoWriter object will create 
# a frame of above defined The output  
# is stored in 'filename.avi' file. 
record=0 # set to 1 to record
if (record):
    result = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (640,480)) 





def BT_connect():
    startTime = time.time()
    # Look for all Bluetooth devices the computer knows about.
    print ("Searching for devices...")
    print ("")
    # Create an array with all the MAC addresses of the detected devices
    nearby_devices = bluetooth.discover_devices()

    # Run through all the devices found and list their name
    num = 0
    print ("Select your device by entering its coresponding number...")
    for i in nearby_devices:
        num+=1
        print (num , ": " , bluetooth.lookup_name( i ))

    # Allow the user to select their bluetooth module. They must have paired it before hand.
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

    result = []
    i=0

    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids,(127,255,0))
        rvec, tvec ,_ = cv2.aruco.estimatePoseSingleMarkers(corners, .15, cameraMatrix, distCoeffs)
        for i in range(0,len(ids)):

            # calc center
            C = np.squeeze(corners[i])

            x1,y1 = C[0] # upper left corner
            x2,y2 = C[2] # lower right corner

            x = int((x1 + x2) / 2)
            y = int((y1 + y2) / 2)

            bearing = calc_heading(rvec[i][0])

            result = [ids[i][0], x, y, bearing]

            i+=1

    return result

def controls(q):
    sock = BT_connect()
    while True:
        phi = q.get()
        print(phi)
        if (phi=="d"):
            print(phi)
            sock.close()
            print("disconnected")
            break

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


# Program to find Closest number in a list 
def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)


def calc_dst(q):

    g,m = SimulatedAnnealing_copy.SA(plot=False,save=True)
    
    i=0
    b=0
    cn=0
    c_n=m[-1]

    import Path_Single_copy
    pa,coords,d = Path_Single_copy.graph_path(m[0],m[1],plot=False,save=False)

    while True:
        
        if i == (len(coords)):

            b+=1
            
            if b == (len(m)-1):
                c_n=g[cn]
                g,m = SimulatedAnnealing_copy.SA(plot=False,save=True)
                b=0
                m.insert(0,c_n)
            
            pa,coords,d = Path_Single_copy.graph_path(m[b],m[b+1],plot=False,save=False)
            i=0

        z1=int(coords[i][0])
        z2=int(coords[i][1])

        ret, frame = cap.read()
        val = aruco_detect(frame)
        if val is not None and len(val) > 0:
            vec1, vec2 = (z1-val[1], z2-val[2])
            theta = math.atan2(vec2,vec1)*180/3.14 # angle between vector(marker->dst) and x-axis
            if theta < 0 :
                theta = 360 + theta


            #cv2.line(frame,(val[1],val[2]), (1200, val[2]), (0,255,100), 2)
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
             
            if mag < 25: # if magnitude is less than 25 update destination
                i+=1 #iterate through list of coordinates


            ## find closest node and print on to screen
            cn=closest_node((val[1],val[2]),g)
            cv2.putText(frame, "closest node {}".format(cn), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3) #font stroke



        cv2.circle(frame, (z1, z2), 5, (255,0,255), -1)
        ## draw nodes on path
        for j in range(len(coords)):
            p1=int(coords[j][0])
            p2=int(coords[j][1])
            cv2.circle(frame, (p1, p2), 5, (255,0,255), -1)

        ## draw nodes on map
        for k in range(len(g)):
            g1=int(g[k][0])
            g2=int(g[k][1])
            cv2.circle(frame, (g1, g2), 5, (0,0,255), 1)

        cv2.imshow('frame', frame)
        
        # Write the frame into the 
        # file 'filename.avi' 
        if (record):
            result.write(frame)
        

        if cv2.waitKey(1) == 27:
            q.put("d")
            break






q=queue.Queue(maxsize=1)
phi=0
t2=Thread(target=controls, args=(q,))
t2.start()
calc_dst(q)
print('All work completed')
t2.join()
print('disconnected')
cap.release()
cv2.destroyAllWindows()

