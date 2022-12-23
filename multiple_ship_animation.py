import pandas as pd
import geopandas as gpd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import cv2

import os
from shapely.geometry import Point, Polygon
import time
from datetime import datetime, timedelta
import math
import argparse

# plt.ion()



def getImage(path):
    return OffsetImage(plt.imread(path))

# imgfilename = 'imgdata/anchor_2.png'
# imgfilename = 'imgdata/pngtree-vector-anchor-material-png-image_798856.jpg'
# imgfilename = 'imgdata/output-onlinepngtools.png'
# imgfilename = 'imgdata/ship_icon.png'
shipimgfilename = 'imgdata/ship_filled_small.png'
anchorimgfilename = 'imgdata/anchor_trans_smaller.png'
# imgfilename = 'imgdata/shipimg.png'

    
parser = argparse.ArgumentParser()
# add hyperparams here
parser.add_argument("--mp4",  type=bool, default=False, help='save as mp4 or display now') 
parser.add_argument("--latlonfile",  default='data/two_ships.csv', help='file with ship location data(AIS)') 
parser.add_argument("--rows_to_read", type=int,  default=1000, help="limit the number of rows from latlonfile") 
parser.add_argument("--interval", type=int,  default=2, help="interval in minutes between each frame")
parser.add_argument("--parent_dir", default='./', help='directory with ENC zone data')
parser.add_argument("--color_on_type",  type=bool, default=True, help='Color ships based on type or status')
parser.add_argument("--video_name", default='maritime', help='video will be media/"video_name"YYYYMMDD_HHmm.mp4')

opt = parser.parse_args()
print(opt)

color_attrib = "VESSEL TYPE" if opt.color_on_type==True else "STATUS"
savemp4 = opt.mp4
vessel_csv_filename=opt.latlonfile
rows_to_read = opt.rows_to_read
interval =opt.interval # interval in minutes
parent_dir = opt.parent_dir
# FLAGS
debug = False
highlight_zones = False # makes things slow TODO: optimise
borderCol = 'black' # alternative : 'black'
frameskip_count = 1	
animation_fps = 20 # TODO: change these numbers as per convenience
mp4_fps = 10

if debug:
	frameskip_count = 1
	animation_fps = 1
	mp4_fps = 2

if(savemp4):
	matplotlib.use("Agg")


vesselcolor = {
	'VESSEL TYPE':{
		# 'Bunkering Tanker':'red',
		# 'Cargo':'blue',
		'Bulk Carrier':'black',
		'Container Ship':'red',
		'Crude Oil Tanker':'chocolate',
		'Oil/Chemical Tanker':'sandybrown',
		'Oil Products Tanker':'peru',
		'other':'blue'
	},
	'STATUS':{
		0:'red',
		1:'blue',
		5:'blueviolet',
		99:'black',
		'other':'yellow'
	}
}
status_info = {

}

dirs = [ x[0] for x in os.walk(parent_dir) ]

# Zone types
achare = []
berths = []
lndare = []
fairwy = []
pilbop = []
tsslpt = []

# setting up Plot
fig, ax = plt.subplots(figsize=(16,9))
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
sea_zone_file = 'TSSLPT.SHP'
jamesMap = True
if jamesMap:
	zoneTypeFileNames =  ['zones.shp', 'landPolygons.shp']
	zoneTypeColors = ['paleturquoise', 'darkseagreen']
	sea_zone_file = 'zones.shp'

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
	print(zoneType)
	colorr = zoneTypeColors[zoneTypeFileNames.index(zoneType)]
	print(f,colorr)
	for zoneFile in zoneFileList:
		gdf = gpd.GeoDataFrame.from_file(zoneFile)
		gdf.plot(ax=ax, color=colorr, edgecolors=borderCol)
		



gpd_shps = []

# for b in berths: TODO berth style is a little different than others. fix it.
# 	gpd.GeoDataFrame.from_file(b).plot(ax=ax, alpha=0.4, marker='.', color='olive', edgecolors='black')


# # Load vessel Data
# vessel_csv_filename='two_ships.csv' #  get it working for non interpolated one
# all_vessels = [pd.read_csv(vessel_csv_filename)]

# total_vessels = len(all_vessels)

# '''
# ENC is the static data
# AIS is the ship movement data
# '''

# lons = []
# lats = []
# heading = []
# clock_time = []
# n_frames = 0
# print("MMSI")
# for vessel in all_vessels:
# 	print(vessel["MMSI"])
# 	temp_lons = list(vessel["LON"])
# 	lons.append(temp_lons)
	
# 	temp_lats = list(vessel["LAT"])
# 	lats.append(temp_lats)
	
# 	temp_heading = list(vessel["HEADING"])
# 	heading.append(temp_heading)

# 	temp_time = list(vessel["TIMESTAMP"])
# 	clock_time.append(temp_time)

# 	if len(vessel) > n_frames:
# 		n_frames = len(vessel)

###-------------------------------------------------------

timestamp_format = '%Y-%m-%d %H:%M:%S'

#=======================#
# prev nearest function for endtime
def prevnear(timestamp, interval):
	newminute = timestamp.minute - timestamp.minute%interval
	return timestamp.replace(second=0, minute=newminute)

# next nearest function for startime
def nextnear(timestamp, interval):
	return prevnear(timestamp + timedelta(minutes=interval), interval)

#find the best latlong value given time t and a set of latlong,t pairs.
def interpolate(arr, t): # TODO: optimise
	#left[0]	 t	  right[0]
	# print(left)
	# print(t)
	# print(right)
	# print()
	r = 0
	for i in range(len(arr)):
		if(arr[i][0]>=t):
			r = i
			break
	# print(l)
	left = arr[r-1]
	right = arr[r]

	dlat = right[1] - left[1]
	dlon = right[2] - left[2]
	

	dx = right[0] - left[0]
	x = t-left[0]
	
	lat = left[1] + x/dx * dlat
	lon = left[2] + x/dx * dlon

	# dh = right[3] - left[3]
	# h = x/dx * dh 
	#rotate the other way if the roation is greator than 180 degrees
	# if(abs(dh)>180):
	# 	h = (h+180)%360

	#alternate heading
	althead = 90 if dlat>0 else 270
	if(dlon!=0):
		althead = math.atan(dlat/dlon) * 360/2/math.pi
	if(dlon < 0):
		althead = (althead+180) %360
	heading = left[3] if (left[3]!=511 and left[3]!=None) else althead #TODO: you might want to use left[3]+h 
	# print(left[1] + lat, left[2] + lon)
	# sys.exit()
	# print("left[3]", left[3])
	return  lat, lon, heading, left[4], left[5]

#=======================#
def getVesselColor():
	if (data[color_attrib][i] in vesselcolor[color_attrib]):
		return vesselcolor[color_attrib][data[color_attrib][i]]
	return vesselcolor[color_attrib]['other']

data = pd.read_csv(vessel_csv_filename)
datalen = min(rows_to_read, len(data["MMSI"]))
data = data.replace({np.nan:None})

exact_starttime = datetime.strptime(data["TIMESTAMP"][0], timestamp_format) 
exact_endtime = datetime.strptime(data["TIMESTAMP"][datalen-1], timestamp_format) 

starttime = nextnear(exact_starttime, interval) 
endtime = prevnear(exact_endtime, interval) 

# get {mmsi:[timestamp1,timestamp2]} for vessel
vessels = {}
for i in range(datalen):
	mmsi = data["MMSI"][i]
	time_obj = datetime.strptime(data["TIMESTAMP"][i], timestamp_format)
	rowvalue = (time_obj, data["LAT"][i] , data["LON"][i], data["HEADING"][i], getVesselColor(), data["STATUS"][i])
	if(mmsi in vessels):
		vessels[mmsi].append(rowvalue)
	else:
		vessels[mmsi]=[rowvalue]

# populate the [[{mmsi:(lat,lon,heading)}, {mmsi:(lat,lon,heading)}]]
# populate the [clock_time										  ]	 for above

interpolatedLatlongs = []
timesteps = []

timestep = starttime
while timestep != endtime:
	vessel_timestep_map = {}
	for mmsi, vesseltimes in vessels.items():
		if (timestep > vesseltimes[0][0] and timestep < vesseltimes[-1][0]):
			lat, lon, heading, col, status =interpolate(vesseltimes, timestep)
			vessel_timestep_map[mmsi] = (lat, lon, heading, col, status) # TODO: add heading
	interpolatedLatlongs.append(vessel_timestep_map)
	timesteps.append(timestep)

	timestep = timestep + timedelta(minutes=interval)
# use timesteps and corresponding interpolatedlatlongs from here
###-----------------------------------------------------------------------
n_frames = len(timesteps)

# patch = patches.FancyArrow(lons[0][0], lats[0][0], lons[0][0], lats[0][0], width=0.005, head_width=0.005, head_length=0.005, color='black')
patch = None
#TODO: where to place this
time_text = plt.text(103.54,1.03, "Current_time")

def dist(x1, y1, x2, y2): # Euclidean distance
	return np.sqrt((x1-x2)**2 + (y1-y2)**2)
	

style="Fancy, head_length=2.0, head_width=4.0, tail_width=0.4"

def init(): 
	return []

startTime = time.time()



def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), borderValue=(255,255,255))


"""
Function to update the frame for each time step.
"""
def update(frame):
	# plt.clf() 
	print("frame %d: %.2f"%(frame, time.time()-startTime))
	frame = frame * frameskip_count	# To speed up the animation

	global patch
	global time_text
	global ax
	
	[p.remove() for p in reversed(ax.patches)]
	[p.remove() for p in reversed(ax.artists)]

	# for child in ax.get_children():
	# 	if isinstance(child, matplotlib.text.Annotation):
	# 		child.remove()
	arrowScale = 0.008
	ship_cur_positions = []
	time_text.set_text(timesteps[frame])
	for mmsi, vessel in interpolatedLatlongs[frame].items():
		# print("vessel: ", vessel)
		latitude = vessel[0]
		longitude = vessel[1]
		heading = vessel[2]/360 * 2 * math.pi
		
		#TODO: change the direction here.. Get the heading from other point
		
		if frame == 0 or mmsi not in interpolatedLatlongs[frame-1]:
			heading = vessel[2]/360 * 2 * math.pi
			patch = patches.FancyArrowPatch((longitude - arrowScale * math.cos(heading), latitude - arrowScale * math.sin(heading)), (longitude, latitude), arrowstyle=style, color='red') #TODO: color could be vessel[3]
		else:
			vessel_prev_pos = interpolatedLatlongs[frame-1][mmsi]
			latdiff = latitude - vessel_prev_pos[0]
			londiff = longitude - vessel_prev_pos[1]
			difflength = math.sqrt(latdiff**2 + londiff**2)
			if(difflength == 0):
				difflength = 1.0
			# heading = math.atan(latdiff/londiff) if londiff != 0 else math.pi/2 # 90 degrees if londiff = 0

			patch = patches.FancyArrowPatch((longitude - arrowScale * londiff /difflength, latitude - arrowScale * latdiff /difflength), (longitude, latitude), arrowstyle=style, color='red') #TODO: color could be vessel[3]
		
		status = vessel[4]
		
		if(int(status) not in [1,5]): # [0, 8, 99 ] means running
			plt.gca().add_patch(patch)
		else:
			imgfilename = shipimgfilename if int(status) not in [1,5 ]  else anchorimgfilename

		# image = cv2.imread(imgfilename)

		# rotated_image = rotate_bound(image, 78)
		# zoom = 1
		# im = OffsetImage(image, zoom=zoom)

		# TODO: Add if makes sense
			ab = AnnotationBbox(getImage(imgfilename), (longitude, latitude), frameon=False)
			ax.add_artist(ab) 

	# Calcuate new longitude and new Latitude and plot it.
	# for i in range(total_vessels):
	# 	if frame+1 < len(lons[i]) : # If animation in progress
	# 		d_xy = dist(lons[i][frame], lats[i][frame], lons[i][frame+1], lats[i][frame+1])
	# 		del_x = del_y = 0.0

	# 		if d_xy == 0.:
	# 			del_x = np.sin(heading[i][frame+1] * np.pi/180.) 
	# 			del_y = np.cos(heading[i][frame+1] * np.pi/180.) 
	# 		else:
	# 			del_x = (lons[i][frame+1] - lons[i][frame])/d_xy
	# 			del_y = (lats[i][frame+1] - lats[i][frame])/d_xy
			
	# 		new_lon = lons[i][frame+1] + 0.012 * del_x
	# 		new_lat = lats[i][frame+1] + 0.012 * del_y
	# 		time_text.set_text(clock_time[i][frame+1])
	# 		patch = patches.FancyArrowPatch((lons[i][frame], lats[i][frame]), (new_lon, new_lat), arrowstyle=style)
	# 		# plt.gca().add_patch(patch)

	# 		ship_cur_positions.append(Point(new_lon,new_lat))
	
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
	highlight_custom_zones= False
	if highlight_custom_zones:
		shp_pnts = gpd.GeoDataFrame(geometry=ship_cur_positions)
		
		pivot_point= (103.8, 1.15)
		diff_point = (0.065,0.0325)
		vert_diff = 0.01
		for i in range(3):
			for j in range(4):
				polygon_points = [(pivot_point[0]+ j* diff_point[0], pivot_point[1]+vert_diff * i + j * diff_point[1])  
					,(pivot_point[0]+ (j+1)* diff_point[0], pivot_point[1]+vert_diff * i + (j+1) * diff_point[1]) 
					,(pivot_point[0]+ (j+1)* diff_point[0], pivot_point[1]+vert_diff * (i+1) + (j+1) * diff_point[1]) 
					,(pivot_point[0]+ (j)* diff_point[0], pivot_point[1]+vert_diff * (i+1) + (j) * diff_point[1]) 
					,(pivot_point[0]+ j* diff_point[0], pivot_point[1]+vert_diff * i + j * diff_point[1])  
					]
				lonngs = [x[0] for x in polygon_points]
				latts = [x[1] for x in polygon_points]
				# print(polygon_points)
				poly = Polygon(zip(lonngs, latts))
				custom_polygon = gpd.GeoDataFrame(index=[0], crs=None, geometry=[poly])
				col = 'wheat'
				# if(i==2 and j==1):
				# 	col ='red'
				for polygon_key, value in enumerate(custom_polygon.geometry.values):
						vals = shp_pnts.within(value)
						for pnt_key, is_inside in enumerate(vals):
							if(is_inside):
								col = 'green'
				
				custom_polygon.plot(ax=ax, color=col, edgecolors=borderCol)


	return []

'''
MAIN
'''
n_frames =n_frames // frameskip_count
print("n_frames: ", n_frames)
if debug:
	n_frames = 20

ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=False, repeat=False, interval = 1000/animation_fps)
handles = [mpatches.Patch(color=col, label=description) for col, description in zip(zoneTypeColors, zoneTypeFileNames)]
handles = handles + [mpatches.Patch(color=col, label=description) for description,col in vesselcolor[color_attrib].items()]
# plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', title="legend", borderaxespad=0.,handles=handles)
	
if savemp4:
	ani.save('media/'+opt.video_name+datetime.now().strftime('%Y%m%d_%H%M')+'.mp4', fps=mp4_fps, dpi=100, savefig_kwargs = {"bbox_inches": "tight", "pad_inches": 0.0})
else:
	plt.show()
