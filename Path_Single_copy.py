"""
Javier Mancilla

Plot graph from edgelist and nodelist over Map and calcuate shortest path using bellman-ford algorithm
Output shortst path as list

TODO 
- make a function to call externally
- integrate SA for scheduling *add this feature to Main file
"""
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import copy

def graph_path( plot=False ):
    #read edge list
    edges = pd.read_csv('Edgelist_updated.csv')
    #add edges from edgelist
    G = nx.from_pandas_edgelist(edges, 'node1', 'node2', ['distance'])
    #read node list
    nodes = pd.read_csv('nodelist.csv')

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
    path=nx.bellman_ford_path(G,'c1_1','c4_3',weight='distance')
    # print the distance of the path
    distance=nx.bellman_ford_path_length(G,'c1_1','c4_3',weight='distance')
    
    for i in range(len(path)):
        p.append(pos['{}'.format(path[i])])
    
    return(path, p, distance)

pa,posit,d = graph_path(plot=True)
print(pa)
print(posit)
print(d)