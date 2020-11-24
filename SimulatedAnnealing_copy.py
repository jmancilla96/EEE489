"""
Javier Mancilla

Simuated Annealing implementation
TODO
- make a function to call externally

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

def generate_coords(n):
    coords = []

    for i in range(n):
        coords.append( [np.random.uniform(), np.random.uniform()] )
        print( coords[i][0], coords[i][1])
    return coords

def get_dist(a, b):
    return( np.sqrt( abs(a[0] - b[0]) + abs(a[1] - b[1]) ) )
    
def get_total_distance(coords):
    dist = 0
        
    for p1, p2 in zip(coords[:-1], coords[1:]):
        dist += get_dist(p1, p2) # add dist from point to point
        
    dist += get_dist(coords[0], coords[-1]) # add dist from initial to final point
        
    return dist


# Simulated annealing
def SA(T=30,factor=0.99,it1=1000,it2=50,plot=False):
    #nodes, coords = Load_coords()
    #coords = Load_coords()
    m,coords = Load_coords() # m is for loading ful nodelist
    cost0 = get_total_distance(coords)
    #T = 30
    #factor = 0.99
    #T_init = T
    it = []
    cost = []
    t = []
    for i in range(it1):#1000
        print(i,'cost = ', cost0)
        
        T = T*factor
        for j in range(it2):#500
            # Exchange two coordinates and get a new neighbour solution
            r1, r2 = np.random.randint(0, len(coords), size=2)

            temp = coords[r1]
            coords[r1] = coords[r2]
            coords[r2] = temp

            # get new cost
            cost1 = get_total_distance(coords)

            if cost1< cost0:
                cost0 = cost1
            else:
                x = np.random.uniform()
                if x < np.exp((cost0-cost1)/T):
                    cost0 = cost1
                else:
                    temp = coords[r1]
                    coords[r1] = coords[r2]
                    coords[r2] = temp
        
        cost.append(cost0)
        it.append(i)
        t.append(T)     
    if (plot):
        plot_tour(coords, cost, it, t)
    """
    for p1,p2 in zip(coords[:-1], coords[1:]): 
        print([p1[0], p2[0]], [p1[1], p2[1]],'\n')
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b') #draw line between coords


    
    ax2.plot([coords[0][0], coords[-1][0]], [coords[0][1], coords[-1][1]], 'b') # draw line between first and last coord 


    for c in coords:
        ax2.plot(c[0], c[1], 'ro')

    ax3.plot(it, cost, 'b')
    ax4.plot(it, t, 'b')
    """
    #plt.show()
    return (m,coords)    





# Fill the coords
#coords=generate_coords(20)

#uncomment to save coords to csv
#df = pd.DataFrame(coords)
#df.to_csv('coords.csv', index=False)
def Load_coords():
    #df = pd.read_csv('coords.csv')
    df = pd.read_csv('new_nodelist.csv',usecols=['x','y'])
    coords = df.values.tolist()
    # update to load random points from nodelist
    coord =[]
    for _ in range(0,8):
        r = random.randint(0,len(coords)-1)
        print(r)
        coord.append(coords[r])

    #print(coord)
    #n = pd.read_csv('nodelist.csv',usecols=['node'])
    #nodes = n.values.tolist()
    #return(coord)
    return(coords, coord)

# Plot
def plot_tour(coords,cost,it,t):
    #coords = Load_coords()
    fig = plt.figure(figsize=(8.6,6.4))
    plt.subplots_adjust( wspace=0.2,hspace = 0.4)
    ax1 = fig.add_subplot(221)
    ax1.set_title('random path')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    ax2 = fig.add_subplot(222)
    ax2.set_title('path')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    ax3 = fig.add_subplot(223)
    ax3.set_title('Cost (Total distance)')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Cost')

    ax4 = fig.add_subplot(224)
    ax4.set_title('Temperature')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Temperature')

    # plot line between points
    for p1,p2 in zip(coords[:-1], coords[1:]): 
        #print([p1[0], p2[0]], [p1[1], p2[1]])
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b') 

    # draw line between first and last point 
    ax1.plot([coords[0][0], coords[-1][0]], [coords[0][1], coords[-1][1]], 'b') 

    # add points to plot
    for c in coords:
        ax1.plot(c[0], c[1], 'ro')

        for p1,p2 in zip(coords[:-1], coords[1:]): 
            #print([p1[0], p2[0]], [p1[1], p2[1]],'\n')
            ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b') #draw line between coords


    
    ax2.plot([coords[0][0], coords[-1][0]], [coords[0][1], coords[-1][1]], 'b') # draw line between first and last coord 


    for c in coords:
        ax2.plot(c[0], c[1], 'ro')

    ax3.plot(it, cost, 'b')
    ax4.plot(it, t, 'b')
    
    plt.show()
   
#m = SA(plot=True)
#SA(30,0.99,1000,500)
#print(m)

#print(Load_coords())
