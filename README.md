# TODO
- Calibrate raspberry pi camera for better Aruco detection
- Setup autoconnect to BT04-A
- merge test.py to ThreadedPositioning.py (currently using test.py as main file)
- Update nodes and edges to include points of interest and traffic with POI.csv
- Make functions for external use
- Create Main 
- Update capstone.md to include more info


# Important files
- Edgelist.csv 
- nodelist.csv
- Path_Single.py (bellman-ford path between two points) Output: list of coordinates in path to next point in schedule 
- SimulatedAnnealing.py (Simulated Annealing implementation) Output: list - schedule of coordinates 
- ThreadedPositioning.py (tracks and controls robot with ArUco markers)

## Structure and data path 
- Output from SimulatedAnnealing.py-> Path_single.py -> test.py


# Requirements
- Matplotlib
- Numpy
- Opencv-contrib-python
- Networkx
- Pybluez
- pandas


# To run
For testing on personal machine
```sh
python test.py
```
on current raspberry pi
```sh
sudo python3 test.py
```
