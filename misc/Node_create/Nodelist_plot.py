"""
Javier Mancilla

Plot nodelist over Map

Requires image of map and nodelist


Uncomment lines 30-40 to plot nodelist over map

(Ignore)
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# creat plot figure with labels
fig = plt.figure()
#plt.subplots_adjust( wspace=0.2,hspace = 0.4)
ax1 = fig.add_subplot(221)
ax1.set_title('random path')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

# plot map image
img = plt.imread("map1.jpg") # change to location of map if needed(may need to copy image in to same folder)
#ax1.imshow(img, extent=[0,1,0,1.1])
ax1.imshow(img)

"""
# load npdelist
#df = pd.read_csv('nodelist.csv')
df = pd.read_csv('new_nodelist.csv')
coords = df.values.tolist()
print(coords)

# add points to plot
for c in coords:
    ax1.plot(c[1], c[2], 'ro')
"""

plt.show()