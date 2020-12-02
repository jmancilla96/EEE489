import bluetooth
import time

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
        if (bluetooth.lookup_name( i ) == "BT04-A"):
            bd_addr = nearby_devices[i]

    #Allow the user to select their bluetooth module. They must have paired it before hand.
    #selection = int(input("> ")) - 1
    #print ("You have selected", bluetooth.lookup_name(nearby_devices[selection]))
    #bd_addr = nearby_devices[selection]

    port = 1
    sock = bluetooth.BluetoothSocket( bluetooth.RFCOMM )
    sock.connect((bd_addr, port))
    return(sock)

sock= BT_connect()