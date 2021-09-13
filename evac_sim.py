#cell 0
from PIL import Image, ImageDraw
import networkx as nx
import numpy as np
from numpy import genfromtxt
import itertools
import pickle
import random
import sys

# Objects required

# map object
# passenger object
# graph object
# simulation engine object

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

'''
    The sim class has three main components:
     1) Aircraft floorplan - to define init state
     2) Graph network - to calculate occupant paths
     3) Occupancy map - to track occupants current position

     
'''
class Floorplan:
    
    def init_example_floorplan():
        '''Generate example floorplan'''
        return np.array([
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

    def init_floorplan(floorplan_file, rotate = False):
        '''
        Create floorplan (numpy array) from the supplied file.
        Rotate will rotate the supplied floorplan by 90deg, so that the left side becomes the top.
        '''
        if rotate:
            return genfromtxt(floorplan_file, delimiter=',')
        else:
            return np.transpose(genfromtxt(floorplan_file, delimiter=','))

class Graph:
    '''
    Class for the graph network which defines the connections between discrete positions.

    '''

    # map reflecting the floorplan schema given above. Used to map floorplan int values to actual element names
    node_map = {0: "Path",
               1: "Wall",
               2: "Seating",
               3: "Impasse",
               4: "Exit"}

    # Types of floorplan elements
    node_types = ['Path','Wall','Seating','Impasse','Exit']
    
    # Subset of floorplan elements which occupants cannot pass through.
    barriers = ["Wall", "Impasse"]

    def __init__(self, floorplan):    
        
        self.movement_difficulty_map = self.init_movement_difficulty_map()
        self.graph = self.init_graph_network(floorplan)

    def init_movement_difficulty_map(self):

        # Mapping of the movement difficulty for each direction and florrplan element
        movement_difficulty = {}

        # Create movement difficulty map - this should be in a config file
        for direction in ['up','side','down']:
            movement_difficulty[direction] = {}
            for type in self.node_types:
                movement_difficulty[direction][type] = {}
                for t in self.node_types:
                    if (type or t) in self.barriers:
                        movement_difficulty[direction][type][t] = 1000000
                    else:
                        movement_difficulty[direction][type][t] = 1
        
        # Path edge cases
        movement_difficulty['up']['Path']['Seating'] = 100
        movement_difficulty['side']['Path']['Seating'] = 5
        movement_difficulty['down']['Path']['Seating'] = 3

        # Exit edge cases
        movement_difficulty['up']['Exit']['Seating'] = 100
        movement_difficulty['side']['Exit']['Seating'] = 5
        movement_difficulty['down']['Exit']['Seating'] = 3

        # Seating edge cases
        movement_difficulty['up']['Seating']['Path'] = 2
        movement_difficulty['side']['Seating']['Path'] = 5
        movement_difficulty['down']['Seating']['Path'] = 100

        movement_difficulty['up']['Seating']['Exit'] = 2
        movement_difficulty['side']['Seating']['Exit'] = 5
        movement_difficulty['down']['Seating']['Exit'] = 100

        movement_difficulty['up']['Seating']['Seating'] = 100
        movement_difficulty['side']['Seating']['Seating'] = 5
        movement_difficulty['down']['Seating']['Seating'] = 100

        return movement_difficulty

    def init_graph_network(self, floorplan):
        G = nx.Graph()
        for i in range(len(floorplan)):
            for j in range(len(floorplan[i])):
                k = (i,j)
                G.add_node(k, type = nodeMap[floorplan[i][j]], pos = (j,len(floorplan)-i))
        
        for i in range(len(floorplan)):
            for j in range(len(floorplan[i])):
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
        return G

class Occupancy():

    def init

#cell 4
def initOccupancy(a):
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


#cell 15

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

if __name__ == '__main__':

    #filenames
    base = sys.argv[1]
    if 'A380' in base:
        horizontalFlag = True
    else:
        horizontalFlag = False
    mazeFilename = base + '.csv'
    pklFile = base + '.pkl'
    floorplan = base + '.png'
    floorplanPop = base + '_with_pop.png'
    gifName = base + '.gif'

    pklFlag = False
    filename = pklFile

    try:
        flag = bool(sys.argv[2])
    except IndexError:
        flag = False

    if not pklFlag:
        # Initialise MAZE
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

    zoom = 20 # global
    borders = 6 # global
    im = draw_matrix(a, zoom, borders) # global

    im.save(floorplan,)

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


