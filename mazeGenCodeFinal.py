#cell 0
from PIL import Image, ImageDraw
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from ast import literal_eval as make_tuple
from numpy import genfromtxt
import itertools
import pickle
import copy
import random

#cell 1
## Initialisation Functions

#cell 2
# Define 'maze'

'''
    Schema:

    0: Unobstructed
    1: Wall
    2: Seating
    3: Impasse
    4: Exit    
'''
    
def initTestMaze():

    a = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [4 ,0, 0, 0, 0, 0, 0, 0, 1],
        [4 ,0, 0, 0, 0, 2, 2, 2, 1],
        [1, 2, 2, 2, 0, 3, 3, 3, 1],
        [1, 0, 0, 0, 0, 3, 3, 3, 1],
        [1, 2, 2, 2, 0, 3, 3, 3, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 2, 2, 2, 0, 2, 2, 2, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ])
    
    return a

def initMaze(file='floorPlanA380.csv', horizontalFlag = False):
  
    a = genfromtxt(file, delimiter=',')
    
    if horizontalFlag:
        return np.transpose(a)
    else:
        return a
    

#cell 3
# Create graph network from 'maze'
def initGraph(maze):
    # Node type dict
    nodeMap = {0: "Path",
               1: "Wall",
               2: "Seating",
               3: "Impasse",
               4: "Exit"}
    
    # Define edge logic
    types = ['Path','Wall','Seating','Impasse','Exit']
    edgeMap = {}

    barriers = ["Wall", "Impasse"]

    for d in ['up','side','down']:
        edgeMap[d] = {}
        for ty in types:
            edgeMap[d][ty] = {}
            for t in types:
                if (ty in barriers) or (t in barriers):
                    val = 1000000
                else:
                    val = 1      
                edgeMap[d][ty][t] = val

    # Path edge cases
    edgeMap['up']['Path']['Seating'] = 100
    edgeMap['side']['Path']['Seating'] = 5
    edgeMap['down']['Path']['Seating'] = 3

    # Exit edge cases
    edgeMap['up']['Exit']['Seating'] = 100
    edgeMap['side']['Exit']['Seating'] = 5
    edgeMap['down']['Exit']['Seating'] = 3

    # Seating edge cases
    edgeMap['up']['Seating']['Path'] = 2
    edgeMap['side']['Seating']['Path'] = 5
    edgeMap['down']['Seating']['Path'] = 100

    edgeMap['up']['Seating']['Exit'] = 2
    edgeMap['side']['Seating']['Exit'] = 5
    edgeMap['down']['Seating']['Exit'] = 100

    edgeMap['up']['Seating']['Seating'] = 100
    edgeMap['side']['Seating']['Seating'] = 5
    edgeMap['down']['Seating']['Seating'] = 100
    
    G = nx.Graph()
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            k = (i,j)
            G.add_node(k, type = nodeMap[maze[i][j]], pos = (j,len(maze)-i))
    
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            k = (i,j)
            sourceType = G.nodes[k]['type']
            # Assign upwards edge
            if i > 0:
                l = (i-1,j)
                targetType = G.nodes[l]['type']
                w =  edgeMap['up'][sourceType][targetType] # Edge weight
                G.add_edge(k,l,weight = w)
            # Assign right edge
            if j < (len(a[i])-2):
                l = (i,j+1)
                targetType = G.nodes[l]['type']
                w =  edgeMap['side'][sourceType][targetType] # Edge weight
                G.add_edge(k,l,weight = w)
            # Assign left edge
            if j > 0:
                l = (i,j-1)
                targetType = G.nodes[l]['type']
                w =  edgeMap['side'][sourceType][targetType] # Edge weight
                G.add_edge(k,l,weight = w)
            # Assign downwards edge
            if i < (len(a)-2):
                l = (i+1,j)
                targetType = G.nodes[l]['type']
                w =  edgeMap['down'][sourceType][targetType] # Edge weight
                G.add_edge(k,l,weight = w)
    return G, nodeMap, edgeMap

#cell 4
def initOccupancy(maze):
    # Initialise occupancy grid. Inidividuals spawned into all seat locations.
    # Individual IDs assigned on encounter
    
    # Note maze is a numpy array so is passed by REFERENCE - modification WILL change
    # Py default is pass by ASSIGNMENT
    
    occupancy = np.zeros(a.size).reshape(a.shape)
    
    individualID = 1
    
    for i in range(len(a)):
        for j in range(len(a[i])):
            if a[i,j] == 2:
                occupancy[i,j] = individualID
                individualID += 1
    
    return occupancy

#cell 5
def initPopulation(graph, occupancy):
    population = []
    for i in range(len(occupancy)):
        for j in range(len(occupancy[i])):
            if occupancy[i,j] > 0:
                # Get initial path selection
                # No need to pass G, but included for clarity
                path = getPath(graph, (i,j))
                
                # Append individual to population with default state
                population.append({'ID': int(occupancy[i,j]),
                                   'location': (i,j),
                                   'path': path,
                                   'step': 0,
                                   'lastMove': 0,
                                   'lastPathChange': 0,
                                   'moved': False,
                                   'canMove': True,
                                   'escaped': False,
                                   'firstMove': random.randint(5, 20)})
                
    return population

#cell 6
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

#cell 7
def searchNodes(G, attr, val):
    return [x for x,y in G.nodes(data=True) if y[attr]==val]

#cell 8
def getPath(graph, source, pathFlag = False, individual = []):
    ## Given a graph and a source, find the shortest paths and select one for an individual
    
    exits = searchNodes(G, 'type','Exit')
    paths = []
    for exit in exits:
        paths.extend([[item for item in path] for path in itertools.islice(nx.all_shortest_paths(G, source, exit, weight = "weight"),3)])
    
    # Sort paths from shortest to longest
    paths = sorted(paths, key=len)
    # Remove paths of equal length (reduce diversity)
    while len([len(p) for p in paths]) != len(set([len(p) for p in paths])):
        for i, path in enumerate(paths):
            if len(path) == len(paths[i-1]):
                paths.pop(i)
                
    p = []      
    for path in paths:
        for i, l in enumerate(path):
            if (a[l] == 3) or (a[l] == 1) or ((i > 0) and a[l] == 2):
                break
        else:
            p.append(path)
    paths = p

    
    if pathFlag:
        paths = [p for p in paths if p[1] != individual['path'][individual['step'] + 1]]
    
    if len(paths) > 0:
        # Logic to select path
        lenPaths = np.array([len(p) for p in paths]) # Get array of path lengths
        lenPaths = sum(lenPaths) - lenPaths # Invert so shortest paths have greatest value
        pPath = softmax(lenPaths) # Calculate probabilities
        path = paths[np.random.choice(len(paths), 1, p=pPath)[0]] # Select path based on probability
        
    else:
        path = []
        
    return path

#cell 9
## Simulation Functions

#cell 10
def moveIndividual(a, G, occ, item, i, t, nodeMap, edgeMap):
    
    movedFlag = False
    if t > 50:
        moveTimeLimit = 1
    else:
        moveTimeLimit = 5
    
    if t < 15:
        item['lastMove'] = t

    if item['moved']:

        item['canMove'] = False
    
    else:
        
        if occ[item['path'][item['step'] + 1]] > 0:
            item['canMove'] = False
            if t > item['firstMove']:
                    if (t-item['lastMove']) > moveTimeLimit:
                       
                        newPath = updatePath(G, item, occ, a)
                        if len(newPath) > 0:
                            item['path'] = newPath
                            item['step'] = 0

        elif t > item['firstMove']:
            if a[item['path'][item['step'] + 1]] == 4:
                if t < 11.1:
                    item['canMove'] = False
                    item['moved'] = True
                    item['lastMove'] = t
                    movedFlag = True
                else:
                    item['canMove'] = False
                    item['moved'] = True
                    item['lastMove'] = t
                    movedFlag = True
                    # Remove current position from occupancy
                    occ[item['location']] = 0
                    item['step'] += 1
                    item['location'] = item['path'][item['step']] # Update location
                    occ[item['location']] = 0 
                    item['escaped'] = True

    
            else:
                # If individual NOT blocked
                item['canMove'] = False
                item['moved'] = True
                item['lastMove'] = t
                movedFlag = True

                # Remove current position from occupancy
                occ[item['location']] = 0
                item['step'] += 1
                item['location'] = item['path'][item['step']] # Update location

                occ[item['location']] = item['ID'] 

                G = updateGraph(item['location'], a, occ, G, nodeMap, edgeMap)           
        else:
            item['canMove'] = False
    return movedFlag

#cell 11
def updatePath(G, item, occ, a):
    # Is there a free adjacent square?
    freeMoves = False
    loc = item['location']
    for i,j in [(0,-1), (1,0), (-1,0), (0,1)]:
        target = ((loc[0] + i), (loc[1] + j))
        if (occ[target] == 0) and (a[target] == 0):
            freeMoves = True
            break
    if freeMoves:
        path = getPath(G, loc, True, item)
    else:
        path = []
            
    return path

#cell 12
def updateGraph(source, a, occ, G, nodeMap, edgeMap):
    
    dirMap = {(0,-1): 'side',
              (1,0): 'up',
              (-1,0): 'down',
              (0,1): 'side'}
    
    for i,j in [(0,-1), (1,0), (-1,0), (0,1)]:
        target = (source[0] + i, source[1] + j)
        
        c = (0,0)
        b = a.shape
        
        if all([(c < target) for c, target in zip(c,target)]) and \
            all([(target < b) for target, b in zip(target,b)]):
            # Update based on MAZE
            sourceType = nodeMap[a[source]]
            targetType = nodeMap[a[target]]
            mazeWeight = edgeMap[dirMap[(i,j)]][sourceType][targetType]

            # Update based on OCCUPANCY
            occWeight = 1000 if occ[target] > 0 else 0
            weight = max(mazeWeight, occWeight)
            #print(f'Source: {source}')
            #print(f'Target: {target}')
            #print(f'Weight: {weight}')
            G[source][target]['weight'] = weight
            
    return G


#cell 13
## Initialise simulation



#cell 14
#filenames
base = 'BWB_restricted_exit'
if 'A380' in base:
    horizontalFlag = True
else:
    horizontalFlag = False
mazeFilename = base + '.csv'
pklFile = base + '.pkl'
floorplan = base + '.png'
floorplanPop = base + '_with_pop.png'
gifName = base + '.gif'

#cell 15
# Initialisation process can be longest part. Pickle setup for speed.
pklFlag = False
filename = pklFile

if not pklFlag:
    # Initialise MAZE
    flag = True
    if flag:
        a =initMaze(mazeFilename, horizontalFlag) # global
    else:
        a = initTestMaze() # global

    # Initialise GRAPH
    G, nodeMap, edgeMap = initGraph(a) # global (NOTE: GRAPH object passed by REFERENCE by default)
    # Hence okay not to pass G between functions as would be edited as if global anyway

    # Initialise OCCUPANCY GRID
    occ = initOccupancy(a)

    # Initialise POPULATION
    pop = initPopulation(G, occ)
    
    with open(filename, 'wb') as f: 
        pickle.dump([a, G, occ, pop, nodeMap, edgeMap], f)

else:
    with open(filename, 'rb') as f:
        a, G, occ, pop, nodeMap, edgeMap = pickle.load(f)
        
timestep = 1/2
endFlag = False
t = 0

#cell 16
## Utilities

#cell 17
# Draw image of base 'maze'
def draw_matrix(a, zoom, borders):
    im = Image.new('RGB', (zoom * len(a[0]), zoom * len(a)), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    for i in range(len(a)):
        for j in range(len(a[i])):
            color = (255, 255, 255)
            r = 0
            if a[i][j] == 1:
                color = (0, 21, 36)
            elif a[i][j] == 2:
                color = (21, 97, 109)
            elif a[i][j] == 3:
                color = (181, 182, 130)
            elif a[i][j] == 4:
                color = (255, 125, 0)
            draw.rectangle((j*zoom+r, i*zoom+r, j*zoom+zoom-r-1, i*zoom+zoom-r-1), fill=color)
    return im

a = initMaze(mazeFilename, horizontalFlag) # global

zoom = 20 # global
borders = 6 # global
im = draw_matrix(a, zoom, borders) # global

im.save(floorplan,)

#cell 18
def drawPop(occ, im):
    draw = ImageDraw.Draw(im)
    zoom = 20
    borders = 6
    for i in range(len(occ)):
        for j in range(len(occ[i])):
            if occ[i,j] > 0:
                r = borders
                draw.ellipse((j * zoom + r, i * zoom + r, j * zoom + zoom - r - 1, i * zoom + zoom - r - 1),
                               fill=(255,0,0))
                
    return im

a = initMaze(mazeFilename, horizontalFlag) # global

zoom = 20 # global
borders = 6 # global
im = draw_matrix(a, zoom, borders) # global
im = drawPop(occ, im)

im.save(floorplanPop,)

#cell 19
## Simulation

#cell 20

# Init gif
baseIm = draw_matrix(a, 20, 6)
images = []

while not endFlag:    
    
    # At start of each iteration noone has moved and everyone can move
    for i in pop:
        i['moved'] = False
        i['canMove'] = True
    
    # Draw population
    images.append(drawPop(occ, baseIm.copy()))
    
    # Loop until no population moves are possible (blocked or have moved)
    while any([m['canMove'] for m in pop]):
        
        for i, item in enumerate(pop):
            
            movedFlag = moveIndividual(a, G, occ, item, i, t, nodeMap, edgeMap)
            
            if movedFlag:
                for m in pop:
                    m['canMove'] = True
                # break
            
        pop = [p for p in pop if p['escaped'] != True]      
   
    
    images.append(drawPop(occ, baseIm.copy()))
    print(f'Timestep: {t} \t Population: {len(pop)}')
    t += timestep
    
    if len(pop) == 0:
        endFlag = True
    
print(f"Time taken: {t} seconds")

images[0].save(gifName,
               save_all=True, append_images=images[1:],
               optimize=False, duration=333, loop=0)


