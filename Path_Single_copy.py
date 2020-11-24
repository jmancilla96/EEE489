"""
Javier Mancilla

Plot graph from edgelist and nodelist over Map and calcuate shortest path using bellman-ford algorithm
Output shortst path as list

TODO 
- make a function to call externally ***Completed
- integrate SA for scheduling *add this feature to Main file
"""
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import copy

def coord2Node(G,x,y):
    #select node based on coordinates
    nod = [j for j,k in G.nodes(data=True) if k['pos']==(x,y)]
    print(nod)
    return(nod)

def graph_path(n0,nf, plot=False ):
    #read edge list
    edges = pd.read_csv('new_Edgelist_update.csv')
    #add edges from edgelist
    G = nx.from_pandas_edgelist(edges, 'node1', 'node2', ['distance'])
    #read node list
    nodes = pd.read_csv('new_nodelist.csv')

    # add nodes to graph
    data = nodes.set_index('node').to_dict('index').items()
    G.add_nodes_from(data)

    # add position atribute from (x,y) coordinates
    for n, p in data:
        G.nodes[n]['pos'] = (p['x'],p['y'])
        

    #print(G.nodes(data=True))
    #print(G.edges(data=True))

    pos=nx.get_node_attributes(G,'pos')
    p=[]

    ##
    #select node based on coordinates
    #nod = [x for x,y in G.nodes(data=True) if y['pos']==(0,730)]
    #print(coord2Node(G,0,730))
    #nod = coord2Node(G,0,730)
    node_0 = coord2Node(G,n0[0],n0[1])
    node_f = coord2Node(G,nf[0],nf[1])
    #print(nod)
    ##

    # create plot
    fig, ax1 = plt.subplots()
    nx.draw_networkx_nodes(G, pos, node_color="r", node_size=50, ax=ax1)
    nx.draw_networkx_edges(G, pos, edge_color="b",width=2.0, alpha=0.75, ax=ax1)
    ax1.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    #nx.draw(G,pos)
    plt.gca().invert_yaxis()
    # plot labes
    ax1.set_title('random path')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    # overlay map
    img = plt.imread("map.jpg")
    ax1.imshow(img)
    if (plot):
        plt.show()

    # print out the shortest path using Bellman Ford
    #path=nx.bellman_ford_path(G,'c1_1','c4_3',weight='distance')
    path=nx.bellman_ford_path(G,node_0[0],node_f[0],weight='distance')
    # print the distance of the path
    #distance=nx.bellman_ford_path_length(G,'c1_1','c4_3',weight='distance')
    distance=nx.bellman_ford_path_length(G,node_0[0],node_f[0],weight='distance')
    
    for i in range(len(path)):
        p.append(pos['{}'.format(path[i])])
    
    return(path, p, distance)

"""
test=[[0, 375], [0, 730], [780, 730], [1990, 1130], [2400, 1950], [2400, 1950], [1616, 2220], [1616, 2800]]
t1=test[0] 
t2=test[5]
pa,posit,d = graph_path(t1,t2,plot=False)
#pa,posit,d = graph_path(plot=True)
print(pa)
print(posit)
print(d)
"""