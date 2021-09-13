# bwb-evac-sim
Simulating the evacuation of a Blended Wing Body aircraft.

`evac_sim.py` is run from the command line. 

The first command line argument is the name of the csv file to be used as the floorplan, without any file extension.

The second command line argument should be 'True', unless you want to use the included test file in which case it can be "False" or ommited.

The total time to evacuate is printed to the command line and the resulted gifs are saved in the directory root.

## Example:

To run the simulation on a floorplan CSV titled `A380.csv`, run the command: `python3 evac_sim.py A380 True`

To run the simulation with the included test floorplan, run the command: `python3 evac_sim.py test False`
