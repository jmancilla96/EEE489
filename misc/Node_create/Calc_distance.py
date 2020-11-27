"""
Javier Mancilla

Calculate distance of edges from edgelist and nodelist
(Ignore)

Requires nodelist with coordinates filled in and edgelist without distance column

"""

import numpy as np
import pandas as pd
##
## load nodelist
##
#coords = pd.read_csv('nodelist.csv')
coords = pd.read_csv('new_nodelist.csv')
#c=coords.values[:,1:]
c = coords.values
c = c.tolist()
##
## load Edgelist
##
df = pd.read_csv('Edgelist.csv')
df = df.values.tolist()
#print(df)

#print(c[0])

##
## Calcurate distance between points a and b
##
def get_dist(a, b):
    return( np.sqrt( abs(a[0] - b[0]) + abs(a[1] - b[1]) ) )


##
## create node pairs for edges into list dist
##
dist = []
for i in range(len(df)):
    #print(df[i][0])
    for x in range(len(c)):
        if c[x][0] == df[i][0]:
            p1=(c[x][1],c[x][2])
            dist.append(p1)

        if c[x][0] == df[i][1]:
            p2=(c[x][1],c[x][2])
            dist.append(p2)
        
print("dist list \n",dist)

##
## calcuated distance between node pairs for edges
##
d=[]
for i in range(len(dist)):
    #print(i)
    if i%2==1:
        g=get_dist(dist[i-1],dist[i])
        d.append(g)

print(df)
print(d)

##
## append distances to edgelist
##
for k in range(len(df)):
    df[k].append(d[k])

##
## write appended edgelist to csv
## uncomment once nodelist and edgelist are updated then run
## Output: writes distance column for edgelist
#
#df = pd.DataFrame(df)
#df.to_csv('new_Edgelist_update.csv', index=False)

