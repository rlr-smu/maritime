import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import os
from shapely.geometry import Point
import time

savemp4 = False
debug = False
highlight_zones = False # makes things slow TODO

frameskip_count = 4	
animation_fps = 10 # TODO: change these numbers to convenience
mp4_fps = 15
if debug:
	frameskip_count = 1
	animation_fps = 1
	mp4_fps = 2

	

if(savemp4):
	matplotlib.use("Agg")
parent_dir = '/mnt/c/Users/chetu/work/maritime/encdata/'
# vessel_dir = '/home/kushagra06/Documents/inter_vessels/'

vessel_dir = '/home/kushagra06/Downloads'
dirs = [ x[0] for x in os.walk(parent_dir) ]

achare = []
berths = []
lndare = []
fairwy = []
pilbop = []
tsslpt = []



fig, ax = plt.subplots()
ax.set_xlim(103.535, 104.03)
ax.set_ylim(1.02, 1.32)

plt.plot([103.62, 103.535], [1.32, 1.095], c='black')
plt.plot([103.535, 103.66], [1.095, 1.02], c='black')
plt.plot([103.66, 103.835], [1.02, 1.16], c='black')
plt.plot([103.835, 104.03], [1.16, 1.22], c='black')

# plt.plot([103.62, 103.535], [1.32, 1.10], c='black')
# plt.plot([103.535, 103.66], [1.095, 1.02], c='black')
# plt.plot([103.66, 103.835], [1.02, 1.16], c='black')
# plt.plot([103.835, 104.025], [1.16, 1.22], c='black')
# rect1 = patches.Rectangle((103.75, 1), 0.45, 0.11, facecolor='gray')
# ax.add_patch(rect1)

# rect2 = patches.Rectangle((103.8, 1.11), 0.45, 0.04, facecolor='gray')
# ax.add_patch(rect2)

# rect3 = patches.Rectangle((103.9, 1.15), 0.45, 0.05, facecolor='gray')
# ax.add_patch(rect3)

# rect4 = patches.Rectangle((103.5, 1.30), 1.0, 0.11, facecolor='gray')
# ax.add_patch(rect4)

zoneTypeFileNames =  ['ACHARE.shp', 'LNDARE.shp', 'FAIRWY.shp', 'PILBOP.shp', 'TSSLPT.shp', 'BERTHS.shp']
zoneTypeColors = ['lavender', 'darkseagreen', 'lightpink', 'peru', 'paleturquoise', 'olive']

# need to use kwargs to define all this.
zoneFiles = {}
for d in dirs:
	if d != parent_dir:
		for f in os.listdir(d):
			if(f in zoneTypeFileNames):
				if(f in zoneFiles):
					zoneFiles[f].append(d+'/'+f)
				else:
					zoneFiles[f] = [d+'/'+f]
for (zoneType,zoneFileList) in zoneFiles.items():
	# print(zoneType)
	colorr = zoneTypeColors[zoneTypeFileNames.index(zoneType)]
	# print(f,colorr)
	for zoneFile in zoneFileList:
		gdf = gpd.GeoDataFrame.from_file(zoneFile)
		gdf.plot(ax=ax, color=colorr, edgecolors='black')
			# if f == 'ACHARE.shp':
			# 	achare.append(d+'/'+f)
			# elif f == 'BERTHS.shp':
			#  	berths.append(d+'/'+f)
			# elif f == 'LNDARE.shp':
			# 	lndare.append(d+'/'+f)
			# elif f == 'FAIRWY.shp':
			# 	fairwy.append(d+'/'+f)
			# elif f == 'PILBOP.shp':
			# 	pilbop.append(d+'/'+f)
			# elif f == 'TSSLPT.shp':
			# 	tsslpt.append(d+'/'+f)

# shps = achare + lndare + fairwy + pilbop + tsslpt

# print("Pilot Points")
# for zoneFile in zoneFiles['PILBOP.shp']:
# 	polygons = gpd.GeoDataFrame.from_file(zoneFile)
# 	for polygon_key, value in enumerate(polygons.geometry.values):
# 		print(polygon_key, value)
gpd_shps = []


# for a in achare:
# 	gpd.GeoDataFrame.from_file(a).plot(ax=ax, color='lavender')

# for l in lndare:
# 	gpd.GeoDataFrame.from_file(l).plot(ax=ax, color='darkseagreen')

# for f in fairwy:
# 	gpd.GeoDataFrame.from_file(f).plot(ax=ax, color='lightpink')

# for p in pilbop:
# 	gpd.GeoDataFrame.from_file(p).plot(ax=ax, color='peru')

# for t in tsslpt:
# 	seazone_polygons = gpd.GeoDataFrame.from_file(t)
# 	seazone_polygons.plot(ax=ax, color='paleturquoise')

# for b in berths:
# 	gpd.GeoDataFrame.from_file(b).plot(ax=ax, alpha=0.4, marker='.', color='olive', edgecolors='black')

all_vessels = []

# for f in os.listdir(vessel_dir):
	# all_vessels.append(pd.read_csv(vessel_dir+f))

all_vessels.append(pd.read_csv('inter_vessel_sample_st0.csv'))

total_vessels = len(all_vessels)

# print(all_vessels)
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

# dx = np.zeros(n_frames)
# dy = np.zeros(n_frames)

# for i in range(n_frames):
# 	dx[i] = np.sin((float(heading[i]) * np.pi/180.))
# 	dy[i] = np.cos((float(heading[i]) * np.pi/180.))

# arx = np.zeros(n_frames)
# ary = np.zeros(n_frames)

# for i in range(n_frames):
# 	arx[i] = (lons[i] + dx[i])/0.7
# 	ary[i] = (lats[i] + dy[i])/0.5

patch = patches.FancyArrow(lons[0][0], lats[0][0], lons[0][0], lats[0][0], width=0.005, head_width=0.005, head_length=0.005, color='black')
time_text = plt.text(103.85,1.05, "placehodlr")
#patch = patches.FancyArrowPatch((lons[0], lats[0]), (lons[0], lats[0]))

def dist(x1, y1, x2, y2):
	return np.sqrt((x1-x2)**2 + (y1-y2)**2)
	

style="Fancy, head_length=2.0, head_width=4.0, tail_width=0.4"

def init():
	ax.set_xlim(103.535, 104.03)
	ax.set_ylim(1.02, 1.32)
	
	plt.plot([103.62, 103.535], [1.32, 1.095], c='black')
	plt.plot([103.535, 103.66], [1.095, 1.02], c='black')
	plt.plot([103.66, 103.835], [1.02, 1.16], c='black')
	plt.plot([103.835, 104.03], [1.16, 1.22], c='black')
	# rect1 = patches.Rectangle((103.75, 1), 0.45, 0.11, facecolor='gray')
	# ax.add_patch(rect1)

	# rect2 = patches.Rectangle((103.8, 1.11), 0.45, 0.04, facecolor='gray')
	# ax.add_patch(rect2)

	# rect3 = patches.Rectangle((103.9, 1.15), 0.45, 0.05, facecolor='gray')
	# ax.add_patch(rect3)

	# rect4 = patches.Rectangle((103.5, 1.46), 0.70, 0.11, facecolor='gray')
	# ax.add_patch(rect4)

	# for _ in range(total_vessels):
	# 	if patch in ax.patches:
	# 		ax.patches.remove(patch)

	
	# ax.add_patch(patch)
	
	return []
	# return patch,

startTime = time.time()
def update(frame):

	print("frame %d: %.2f"%(frame, time.time()-startTime))
	# if frame < n_frames:
	frame = frame * frameskip_count	
	global patch
	global time_text
		#ax.patches.remove(patch)
		#patch = patches.FancyArrow(lons[frame], lats[frame], lons[frame+1], lats[frame+1], width=0.01, length_includes_head=True, head_width=0.02, head_length=0.02)
		
		# for _ in range(total_vessels):
		# 	if patch in ax.patches:
		# 		ax.patches.remove(patch)
	[p.remove() for p in reversed(ax.patches)]

	
	ship_cur_positions = []
	for i in range(total_vessels):
		if len(lons[i]) > frame+1:
		# print(frame, len(lons[i]))
			# patches.ArrowStyle("Fancy", head_length=50, head_width=50, tail_width=50)
			# style="simple, head_length=2, head_width=1, tail_width=1"
			# style="Fancy, head_length=3, head_width=2"
			# tan_theta = np.tan(heading[i][frame+1] * np.pi/180.)
			d_xy = dist(lons[i][frame], lats[i][frame], lons[i][frame+1], lats[i][frame+1])

			del_x = del_y = 0.0
			
			# del_x = np.cos(heading[i][frame+5] * np.pi/180.)
			# del_y = np.sin(heading[i][frame+5] * np.pi/180.)

			if d_xy == 0.:
				del_x = np.sin(heading[i][frame+1] * np.pi/180.) 
				del_y = np.cos(heading[i][frame+1] * np.pi/180.) 
			else:
				del_x = (lons[i][frame+1] - lons[i][frame])/d_xy
				del_y = (lats[i][frame+1] - lats[i][frame])/d_xy
			
			new_lon = lons[i][frame+1] + 0.012 * del_x
			new_lat = lats[i][frame+1] + 0.012 * del_y
			time_text.set_text(clock_time[i][frame+1])
			# print(clock_time[i][frame+1])
			# if(dist(lons[i][frame], lats[i][frame], new_lon, new_lat) < 0.008 and lons[i][frame]>=103.5 and lons[i][frame]<=104.1 and lats[i][frame]>=1.05 and lats[i][frame]<=1.36):
			# 	print(lons[i][frame], lats[i][frame])
			# patch = patches.FancyArrow(lons[i][frame], lats[i][frame], lons[i][frame+1]-lons[i][frame], lats[i][frame+1]-lats[i][frame], width=0.005, color='black', head_width=0.005, head_length=0.005, linestyle='--')
			patch = patches.FancyArrowPatch((lons[i][frame], lats[i][frame]), (new_lon, new_lat), arrowstyle=style)
			# patch = patches.Arrow(lons[i][frame], lats[i][frame], lons[i][frame+1]-lons[i][frame], lats[i][frame+1]-lats[i][frame], width=1)
			plt.gca().add_patch(patch)

			# TODO: use new_lat and new_lon to light up the zones.
			ship_cur_positions.append(Point(new_lon,new_lat))
	
	
	# now that you have the color data, just plot the polgons
	# for t in tsslpt:
	# 	seazone_polygons = gpd.GeoDataFrame.from_file(t)

	# 	shp_pnts = gpd.GeoDataFrame(geometry=ship_cur_positions)
	# 	# print("shp_pnts:", shp_pnts)
	# 	seazone_polygons.crs = None
	# 	col = ['lavender'] * len(seazone_polygons.geometry.values) # get the default color here.
	# 	for polygon_key, value in enumerate(seazone_polygons.geometry.values):
	# 			vals = shp_pnts.within(value)
	# 			# print("len(pnts)", len(pnts))
	# 			for pnt_key, is_inside in enumerate(vals):
	# 				if(is_inside):
	# 					col[polygon_key] = 'red'
	# 					# print(key,ii, type(value))
	# 	seazone_polygons.plot(ax=ax, color=col)
	
	if highlight_zones:
		shp_pnts = gpd.GeoDataFrame(geometry=ship_cur_positions)
		# plt.cla()
		for (zoneType,zoneFileList) in zoneFiles.items():
			colorr = zoneTypeColors[zoneTypeFileNames.index(zoneType)]
			# print(f,colorr)
			
			for zoneFile in zoneFileList:
				seazone_polygons = gpd.GeoDataFrame.from_file(zoneFile)

				# seazone_polygons = gpd.GeoDataFrame.from_file(t)

				
				# print("shp_pnts:", shp_pnts)
				seazone_polygons.crs = None
				col = [colorr] * len(seazone_polygons.geometry.values) # get the default color here.
				for polygon_key, value in enumerate(seazone_polygons.geometry.values):
						vals = shp_pnts.within(value)
						# print("len(pnts)", len(pnts))
						for pnt_key, is_inside in enumerate(vals):
							if(is_inside):
								col[polygon_key] = 'red' 	# TODO: if you want the color intensity to represent the number
															# of ships, increment some value here, and decide the color outside loop based on that value
															# TODO: also, 4 for loops here, there's scope for optimisation
								break
							
				seazone_polygons.plot(ax=ax, color=col, edgecolors='black')


				# gdf.plot(ax=ax, color=colorr)

	return []
	# return patch,

print("n_frames: ", n_frames)
if debug:
	n_frames = 10

ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=True, repeat=False, interval = 1000/animation_fps)
if savemp4:
	ani.save('video_new.mp4', fps=mp4_fps)
else:
	plt.show()
# plt.savefig('sg_boundaries.png')

# 103.62, 1.32 
# 103.535, 1.095

# 103.535, 1.095
# 103.66, 1.02

# 103.66, 1.02
# 103.835, 1.16

# 103.835, 1.16
# 104.03, 1.22
