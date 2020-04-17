import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import os
from shapely.geometry import Point, Polygon
import time

# FLAGS
savemp4 = True
debug = False
highlight_zones = False # makes things slow TODO
borderCol = None # alternative : 'black'
frameskip_count = 4	
animation_fps = 10 # TODO: change these numbers as per convenience
mp4_fps = 15
if debug:
	frameskip_count = 1
	animation_fps = 1
	mp4_fps = 2

if(savemp4):
	matplotlib.use("Agg")

parent_dir = '/mnt/c/Users/chetu/work/maritime/encdata/'
dirs = [ x[0] for x in os.walk(parent_dir) ]

# Zone types
achare = []
berths = []
lndare = []
fairwy = []
pilbop = []
tsslpt = []

# setting up Plot
fig, ax = plt.subplots()
ax.set_xlim(103.535, 104.03)
ax.set_ylim(1.02, 1.32)

#draw 4 lines to mark boundary
plt.plot([103.62, 103.535], [1.32, 1.095], c='black')
plt.plot([103.535, 103.66], [1.095, 1.02], c='black')
plt.plot([103.66, 103.835], [1.02, 1.16], c='black')
plt.plot([103.835, 104.03], [1.16, 1.22], c='black')

#Zone info input from files
zoneTypeFileNames =  ['ACHARE.shp', 'LNDARE.shp', 'FAIRWY.shp', 'PILBOP.shp', 'TSSLPT.shp', 'BERTHS.shp']
zoneTypeColors = ['lavender', 'darkseagreen', 'lightpink', 'peru', 'paleturquoise', 'olive']

#Store full paths of all zonefiles
zoneFiles = {}
for d in dirs:
	if d != parent_dir:
		for f in os.listdir(d):
			if(f in zoneTypeFileNames):
				if(f in zoneFiles):
					zoneFiles[f].append(d+'/'+f)
				else:
					zoneFiles[f] = [d+'/'+f]

# Load the GeoDataframe from files
for (zoneType,zoneFileList) in zoneFiles.items():
	# print(zoneType)
	colorr = zoneTypeColors[zoneTypeFileNames.index(zoneType)]
	# print(f,colorr)
	for zoneFile in zoneFileList:
		gdf = gpd.GeoDataFrame.from_file(zoneFile)
		gdf.plot(ax=ax, color=colorr, edgecolors=borderCol)
		

if debug:
	print("TSSLPT")
	for zoneFile in zoneFiles['TSSLPT.shp']:
		print("\n\n", zoneFile, "\n\n")
		polygons = gpd.GeoDataFrame.from_file(zoneFile)
		for polygon_key, value in enumerate(polygons.geometry.values):
			print(polygon_key, value)
		break
	
	# Testing adding polygon:
	polygon_points = [(103.95, 1.05) ,(104.0, 1.10),(103.95, 1.15),(103.95, 1.05)]
	longs = [x[0] for x in polygon_points]
	lats = [x[1] for x in polygon_points]
	print(polygon_points)
	poly = Polygon(zip(longs, lats))
	custom_polygon = gpd.GeoDataFrame(index=[0], crs=None, geometry=[poly])
	custom_polygon.plot(ax=ax, color='red', edgecolors=borderCol)

gpd_shps = []

# for b in berths: TODO berth style is a little different than others. fix it.
# 	gpd.GeoDataFrame.from_file(b).plot(ax=ax, alpha=0.4, marker='.', color='olive', edgecolors='black')

# Load vessel Data
vessel_csv_filename='inter_vessel_sample_st0.csv'
all_vessels = [pd.read_csv(vessel_csv_filename)]

total_vessels = len(all_vessels)

'''
ENC is the static data
AIS is the ship movement data
'''

lons = []
lats = []
heading = []
clock_time = []
n_frames = 0

for vessel in all_vessels:
	temp_lons = list(vessel["LON"])
	lons.append(temp_lons)
	
	temp_lats = list(vessel["LAT"])
	lats.append(temp_lats)
	
	temp_heading = list(vessel["HEADING"])
	heading.append(temp_heading)

	temp_time = list(vessel["TIMESTAMP"])
	clock_time.append(temp_time)

	if len(vessel) > n_frames:
		n_frames = len(vessel)

patch = patches.FancyArrow(lons[0][0], lats[0][0], lons[0][0], lats[0][0], width=0.005, head_width=0.005, head_length=0.005, color='black')

#TODO: where to place this
time_text = plt.text(103.85,1.05, "Current_time")

def dist(x1, y1, x2, y2): # Euclidean distance
	return np.sqrt((x1-x2)**2 + (y1-y2)**2)
	

style="Fancy, head_length=2.0, head_width=4.0, tail_width=0.4"

def init(): 
	return []

startTime = time.time()

"""
Function to update the frame for each time step.
"""
def update(frame): 
	print("frame %d: %.2f"%(frame, time.time()-startTime))
	frame = frame * frameskip_count	# To speed up the animation

	global patch
	global time_text

	[p.remove() for p in reversed(ax.patches)]

	ship_cur_positions = []

	# Calcuate new longitude and new Latitude and plot it.
	for i in range(total_vessels):
		if frame+1 < len(lons[i]) : # If animation in progress
			d_xy = dist(lons[i][frame], lats[i][frame], lons[i][frame+1], lats[i][frame+1])
			del_x = del_y = 0.0

			if d_xy == 0.:
				del_x = np.sin(heading[i][frame+1] * np.pi/180.) 
				del_y = np.cos(heading[i][frame+1] * np.pi/180.) 
			else:
				del_x = (lons[i][frame+1] - lons[i][frame])/d_xy
				del_y = (lats[i][frame+1] - lats[i][frame])/d_xy
			
			new_lon = lons[i][frame+1] + 0.012 * del_x
			new_lat = lats[i][frame+1] + 0.012 * del_y
			time_text.set_text(clock_time[i][frame+1])
			patch = patches.FancyArrowPatch((lons[i][frame], lats[i][frame]), (new_lon, new_lat), arrowstyle=style)
			plt.gca().add_patch(patch)

			ship_cur_positions.append(Point(new_lon,new_lat))
	
	# whether to highlight the zones the ship is currently in. Slow :(  TODO: make it fast
	if highlight_zones:
		shp_pnts = gpd.GeoDataFrame(geometry=ship_cur_positions)
		for (zoneType,zoneFileList) in zoneFiles.items():
			colorr = zoneTypeColors[zoneTypeFileNames.index(zoneType)]
			# print(f,colorr)
			
			for zoneFile in zoneFileList:
				seazone_polygons = gpd.GeoDataFrame.from_file(zoneFile)
				seazone_polygons.crs = None
				col = [colorr] * len(seazone_polygons.geometry.values) # get the default color here.
				for polygon_key, value in enumerate(seazone_polygons.geometry.values):
						vals = shp_pnts.within(value)
						for pnt_key, is_inside in enumerate(vals):
							if(is_inside):
								col[polygon_key] = 'red' 	# TODO: if you want the color intensity to represent the number
															# of ships, increment some value here, and decide the color outside loop based on that value
															# TODO: also, 4 for loops here, there's scope for optimisation
								break
							
				seazone_polygons.plot(ax=ax, color=col, edgecolors='black')

	return []

'''
MAIN
'''
n_frames =n_frames // frameskip_count
print("n_frames: ", n_frames)
if debug:
	n_frames = 10

ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=True, repeat=False, interval = 1000/animation_fps)
if savemp4:
	ani.save('video_new.mp4', fps=mp4_fps)
else:
	plt.show()
