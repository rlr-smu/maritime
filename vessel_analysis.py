from datetime import datetime, timedelta
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import argparse
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
import sys
import math
import time
from multiprocessing import Pool
import matplotlib.patches as mpatches

# from matplotlib import rc
# import matplotlib.pylab as plt

# rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# rc('text', usetex=True)
# import matplotlib.font_manager as font_manager
# font_manager._rebuild()

matplotlib.rc('font',family='Times New Roman')

parser = argparse.ArgumentParser()
parser.add_argument("--latlonfile",  default='data/two_ships.csv', help='file with ship location data(AIS)') 
parser.add_argument("--rows_to_read", type=int,  default=1000, help="limit the number of rows from latlonfile") 
parser.add_argument("--column_index", type=int,  default=0, help="0. MMSI,VESSEL TYPE,LENGTH,WIDTH,IMO,5.DESTINATION,STATUS,SPEED (KNOTSx10),LON,LAT,10.COURSE,HEADING,TIMESTAMP") 
parser.add_argument("--zone_file_location",  default='/mnt/c/Users/chetu/work/maritime/encdata/jamesZones', help="Single zone file to track traffic") 
parser.add_argument("--exit_entry_diff",  type=int,  default=12, help="difference between entry and exit of a ship from map") 
parser.add_argument("--zone_transit",  type=bool,  default=False, help="Analyse the time taken to cross a zone?") 
parser.add_argument("--trajectory_type_analysis",  type=bool,  default=False, help="check what type a trajectory is(eastbound, other stuff)") 
parser.add_argument("--save_traj_table",  type=bool,  default=False, help="Save the file back after interpolation") 
parser.add_argument("--zone_file",  default='/mnt/c/Users/chetu/work/maritime/encdata/jamesZones/zones.shp', help="Single zone file to track traffic") 

opt = parser.parse_args()
print(opt)
startime = time.time()
vessel_csv_filename=opt.latlonfile
rows_to_read = opt.rows_to_read
interval = 30
timestamp_format = '%Y-%m-%d %H:%M:%S'

data = pd.read_csv(vessel_csv_filename)
datalen = min(rows_to_read, len(data["MMSI"]))
print("datalen", datalen)
data = data.replace({np.nan:None})

columns = ['MMSI', 'VESSEL TYPE', 'LENGTH', 'WIDTH', 'IMO', 'DESTINATION', 'STATUS', 'SPEED (KNOTSx10)', 'LON', 'LAT', 'COURSE', 'HEADING', 'TIMESTAMP']
attribute = columns[opt.column_index]
###-------------------------------------------------------

timestamp_format = '%Y-%m-%d %H:%M:%S'

maxlon = 104.03 # as per quote
minlon = 103.535 # as per quote
maxlon -= 0.05 # increase prob of trajectory breaking
minlon += 0.05 # increase prob of trajectory breaking


vessels = {}
trajectories={}
for i in range(datalen):
	mmsi = data["MMSI"][i]
	if(mmsi in vessels):
		vessels[mmsi].append(i)
	else:
		vessels[mmsi]=[i]

# print(vessels)
# use the above data for whatever purpose-------------------
def totime(timestring):
    return datetime.strptime(timestring, timestamp_format)

def destination(vessels):
    for mmsi, vesselrows in vessels.items():
        dests = set([data['DESTINATION'][rowid] for rowid in vesselrows])
        if(len(dests) > 1):
            print(mmsi, dests,data['IMO'][vesselrows[0]])


#=====================================================================#
# prev nearest function for endtime
def prevnear(timestamp, interval):
	newsecond = timestamp.second - timestamp.second%interval
	return timestamp.replace(second=newsecond)

# next nearest function for startime
def nextnear(timestamp, interval):
	return prevnear(timestamp + timedelta(seconds=interval), interval)

#find the best latlong value given time t and a set of latlong,t pairs.
def interpolate(arr, t): # TODO: optimise
	# for i in range(len(arr)):
	# 	if(arr[i][0]>=t):
	# 		r = i
	# 		break
	#binary search
	high = len(arr)
	low = 0
	while low + 1 <high:
		mid = (high + low)//2
		if(totime(data['TIMESTAMP'][arr[mid]])>=t):
			high = mid
		else:
			low = mid
	# if (high!=r):
	# 	print("binary search wrong", high, r)
	# 	sys.exit()
	left = arr[high-1]
	right = arr[high]

	dlat = data['LAT'][right] - data['LAT'][left]
	dlon = data['LON'][right] - data['LON'][left]
	

	dx = totime( data['TIMESTAMP'][right]) - totime(data['TIMESTAMP'][left] )
	x = t-totime(data['TIMESTAMP'][left] )
	
	if(dx.total_seconds() !=0):
		lat = data['LAT'][left] + x/dx * dlat
		lon = data['LON'][left] + x/dx * dlon
	else:
		lat = data['LAT'][left]
		lon = data['LON'][left]

	status = data['STATUS'][left]
	speed = data['SPEED (KNOTSx10)'][left]
	course = data['COURSE'][left]
	heading = data['HEADING'][left]
	timestamp = t
	return  [status, speed, lon, lat, course, heading, timestamp]


def break_trajectory(vessels):

    vid = 0
    printevery = 1
    for mmsi, vesselrows in vessels.items():
        if(vid % printevery == 0):
            print("breaking vessel %d, time: %.2f" % (vid, time.time()-startime))
            sys.stdout.flush()
        vid+=1

        breakpoints = [0]
        for i in range(1, len(vesselrows)):
            # 1.if vesselrows[i-1] is far back in time from vesselrows[i]
            # 2.and extrapolation from vesselrows[i-2] [i-1] to [i] takes it out of map
            # 3.and extrapolation from vesselrows[i+1] [i] to [i-1] takes it out of map
            # 4.and vesselrows[i-1] and vesselrows[i] are near boundary
            #   then it's a good idea to break trajectory.

            confidence = 0
            #1
            if(totime(data['TIMESTAMP'][vesselrows[i]]) - totime(data['TIMESTAMP'][vesselrows[i-1]]) > timedelta(hours=2)):
                confidence += 1
            
            
            #2
            if(i-2 >= 0):
                diff = data['LON'][vesselrows[i-1]] - data['LON'][vesselrows[i-2]]
                extrapolated = data['LON'][vesselrows[i-1]] + diff
                if(extrapolated > maxlon or extrapolated < minlon): # going east or west respectively
                    confidence +=1
                    # print("departure ",extrapolated, vesselrows[i-1]+2)
            
            #3
            if(i+1 < len(vesselrows)):
                diff = data['LON'][vesselrows[i+1]] - data['LON'][vesselrows[i]]
                extrapolated = data['LON'][vesselrows[i]] - diff
                if(extrapolated > maxlon or extrapolated < minlon): # coming from east or west respectively
                    confidence +=1
                    # print("arrival ",extrapolated, vesselrows[i-1]+2)
            
            #4 is probably not needed if 2 and 3 are satisfied.

            if(confidence >= 3):
                # print(mmsi, vesselrows[i-1]+2, confidence)
                breakpoints.append(i)
        breakpoints.append(len(vesselrows))
        for i in range(len(breakpoints)-1):
            trajectories[str(mmsi)+'_'+str(i)] = vessels[mmsi][breakpoints[i]:breakpoints[i+1]]

def atboundary(lon):
    return True if lon < minlon or lon > maxlon else False

trajcol = ['red'] * 2 + ['slateblue'] * 2 + ['green'] * 2 + ['sandybrown'] * 2

def categorise_trajectory(trajectories):
    traj_freq= [0] *32
    traj_type_sample = [''] * 32
    discarded = 0

    traj_type = {}
    traj_index=0
    for traj_id, trajectory in trajectories.items():
        if(traj_index %1000 ==0):
            print("traj_id:", traj_index)
        traj_index +=1 

        if(len(trajectory) <10):
            discarded +=1
            traj_type[traj_id] = -1
            continue
        traj_type_id = 0

        firstlon = data['LON'][trajectory[0]]
        lastlon = data['LON'][trajectory[-1]]

        if(not atboundary(firstlon)):
            traj_type_id += 16
        if(not atboundary(lastlon)):
            traj_type_id += 8
        if(firstlon < minlon): # arriving from india
            traj_type_id += 4
        if(lastlon < minlon): # leaving to india
            traj_type_id +=2

        all_status_zero = True
        for rowid in trajectory:
            if(data['STATUS'][rowid] != 0):
                all_status_zero = False
                break
        if(not all_status_zero):
            traj_type_id+=1
    
        traj_freq[traj_type_id] +=1 
        traj_type_sample[traj_type_id] = traj_id + ' ' + ('None' if(data[attribute][trajectory[0]] is None) else str(data[attribute][trajectory[0]]) )
        traj_type[traj_id] = traj_type_id
        if(traj_type_id >= 100 and traj_type_id < 60): # very good trajectory, TODO: refine it later
            lats = [ data['LAT'][rowid] for rowid in trajectory]
            lons = [ data['LON'][rowid] for rowid in trajectory]
            plt.plot( lons, lats, color='blue', alpha=1.0, linewidth=1, linestyle='-' if traj_type_id % 2 == 0 else '--')
    print("discarded: ", discarded)
    # print(traj_type)
    return traj_freq, traj_type_sample, traj_type

def interpolate_traj(traj_context):
    mmsi_t, trajectory = traj_context
    first_time = nextnear(totime(data['TIMESTAMP'][trajectory[0]]), interval)
    last_time = prevnear(totime(data['TIMESTAMP'][trajectory[-1]]), interval)
    timestep = first_time
    i=0
    int_rows = []
    while timestep <= last_time:
        interpolated_row=interpolate(trajectory, timestep)
        int_rows.append(interpolated_row)
        
        timestep = timestep + timedelta(seconds=interval)
        i+=1

    events_df = pd.DataFrame(index=np.arange(0, len(int_rows)),  columns = ['STATUS', 'SPEED (KNOTSx10)', 'LON', 'LAT', 'COURSE', 'HEADING', 'TIMESTAMP'])
    for i,int_row in enumerate(int_rows):
        events_df.loc[i] = int_row

    events_df.to_csv('data/all_30_sec_events/'+mmsi_t+'.csv', index=False)
    print("\ttrajectory count %s, time: %.2f" % (str(mmsi_t), time.time()-startime))
        

def save_traj_table(trajectories, traj_type):

    '''save the ship table first'''
    ship_df = pd.DataFrame(columns=['MMSI_TRAJID', 'VESSEL TYPE', 'LENGTH', 'WIDTH', 'IMO', 'DESTINATION', 'TRAJ_TYPE'])
    for i, key in enumerate(trajectories):
        print(key)
        ship_df.loc[i]= [key] + [ data[x][trajectories[key][0]] for x in columns[1:6] ] + [traj_type[key]]
    ship_df.to_csv('data/all_30_sec_mmsi_traj.csv', index=False)
    
    '''save the events table'''
    with Pool() as pool:
        pool.map(interpolate_traj, trajectories.items())

    # tc=0
    # printevery = 1
    # for mmsi_t, trajectory in trajectories.items():
    #     if(tc%printevery == 0):
    #         print("traj ", tc, mmsi_t)
    #         sys.stdout.flush()

    #     # TODO: do interpolation before saving
    #     first_time = nextnear(totime(data['TIMESTAMP'][trajectory[0]]), interval)
    #     last_time = prevnear(totime(data['TIMESTAMP'][trajectory[-1]]), interval)
    #     timestep = first_time
    #     i=0
    #     int_rows = []
    #     while timestep <= last_time:
    #         interpolated_row=interpolate(trajectory, timestep)
    #         int_rows.append(interpolated_row)
            
    #         timestep = timestep + timedelta(seconds=interval)
    #         i+=1
        
    #     if(tc%printevery == 0):   
    #         print("\t before to_csv %d, time: %.2f" % (tc, time.time()-startime))

    #     events_df = pd.DataFrame(index=np.arange(0, len(int_rows)),  columns = ['STATUS', 'SPEED (KNOTSx10)', 'LON', 'LAT', 'COURSE', 'HEADING', 'TIMESTAMP'])
    #     for i,int_row in enumerate(int_rows):
    #         events_df.loc[i] = int_row
    #     # for i, rowid in enumerate(trajectory):
    #     #     events_df.loc[i] = [ data[x][rowid] for x in columns[6:13] ]
    #     events_df.to_csv('data/all_30_sec_events/'+mmsi_t+'.csv', index=False)
    #     if(tc%printevery == 0):
    #         print("\ttrajectory count %d, time: %.2f" % (tc, time.time()-startime))
    #     tc+=1
    
	

        

#-----------------------------------------------------------
fig, ax = plt.subplots(figsize=(16,9))

print("len(trajectories): , ", len(trajectories))

# destination(vessels)
break_trajectory(vessels)

print("done with break_trajectory: %.2f" % (time.time()-startime))
# print(trajectories)
print("len(trajectories): , ", len(trajectories))

# destination(trajectories)
# sys.exit()
traj_freq, traj_type_sample, traj_type = categorise_trajectory(trajectories)
print("len(traj_type , ", len(traj_type))

if(opt.save_traj_table):
    save_traj_table(trajectories, traj_type)

if(opt.trajectory_type_analysis):
    ax.set_xlim(103.535, 104.03)
    ax.set_ylim(1.02, 1.32)
    ax.set_facecolor('azure')
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    

    print("traj_freq")
    for i, val in enumerate(traj_freq):
        print(i, val, traj_type_sample[i])



    zoneFileList = ['zones.shp', 'landPolygons.shp']
    colorrs = ['aliceblue','yellowgreen']
    borderCol='gray'

    tsscol= 'darkslateblue'
    fairwaycol = 'skyblue'
    ancragecol = 'salmon'
    handles = [mpatches.Patch(color=ancragecol, label='Anchorage'),
        mpatches.Patch(color=tsscol, label='TSS'),
        mpatches.Patch(color=fairwaycol, label='Fairway'),
        mpatches.Patch(color='yellowgreen', label='Landmass'),
        mpatches.Patch(color='azure', label='open waters')]

    for zoneFile,colorr in zip(zoneFileList,colorrs):
        gdf = gpd.GeoDataFrame.from_file(opt.zone_file_location+'/'+zoneFile)
        if(zoneFile is 'zones.shp'):
            ll = 10
            polygons = gdf.geometry.values
            poly_count = len(polygons)

            tsspolyids = [x for x in range(20)]
            fairwaypolyids = [x for x in range(20, 30)]
            anchoragepolyids = [x for x in range(30, 48)]
            
            polycolors = ['aliceblue']* poly_count
            # hatches = ['//'] * poly_count
            for i in anchoragepolyids:
                polycolors[i] = ancragecol
                # hatches[i] = '*'
            for i in tsspolyids:
                polycolors[i] = tsscol
            for i in fairwaypolyids:
                polycolors[i] = fairwaycol
            gdf.plot(ax=ax, color=polycolors, edgecolors=borderCol)
        else:
            gdf.plot(ax=ax, color=colorr, edgecolors=borderCol)
    plt.legend(bbox_to_anchor=(0.02, 0.0), loc='upper left', borderaxespad=0., ncol=3, handles= handles, prop={'size':25}, fontsize=10)
    plt.show()

print("len(trajectories): , ", len(trajectories))
print("len(traj_type , ", len(traj_type))
# zone transit time analysis
if(opt.zone_transit):
    '''for a given zone, identify the trajectores going through that zone and calculate time taken'''

    # zones_to_plot = [x for x in range(64)]
    zones_to_plot = [14]
    
    zoneFile = opt.zone_file
    gdf = gpd.GeoDataFrame.from_file(zoneFile)
    polygons = gdf.geometry.values
    poly_count = len(polygons)

    time_diffs = []
    speeds = []
    print("len(trajectories): , ", len(trajectories))
    print("len(traj_type , ", len(traj_type))

    for mmsi, trajectory in trajectories.items():
        # print("mmsi: ", mmsi )
        if(mmsi not in traj_type):
            print("traj id not found", mmsi) #TODO : this line should never get printed. yet it does...
            continue
        if(traj_type[mmsi] != 4 and traj_type[mmsi] != 5 and traj_type[mmsi] != 1): # only looking at 'to china' ships
            continue
        points = []
        for rowid in trajectory:
            latitude = data['LAT'][rowid]
            longitude = data['LON'][rowid]
            points.append(Point(longitude, latitude))
        shp_pnts = gpd.GeoDataFrame(geometry=points)

        for pind in zones_to_plot:
            # print(polygon)
            already_inside = False
            entry_id = -1
            exit_id = -1
            vals = shp_pnts.within(polygons[pind])
            for pnt_key, is_inside in enumerate(vals):
                if(is_inside):
                    if(not already_inside):
                        entry_id = pnt_key
                    already_inside = True
                    exit_id = pnt_key

                elif(already_inside == True):
                    already_inside = False
                    break # TODO: no need to break if taken care of multiple entry to a zone in a trajectory
            if(entry_id !=-1):
                print(points[entry_id], points[exit_id])
            if(entry_id!=-1 and not already_inside):
                speeds.append(data['SPEED (KNOTSx10)'][trajectory[exit_id]]) # appending both entry and exit speeds
                speeds.append(data['SPEED (KNOTSx10)'][trajectory[entry_id]])

                print(mmsi, totime(data['TIMESTAMP'][trajectory[entry_id]]), totime(data['TIMESTAMP'][trajectory[exit_id]]))
                time_diff = (totime(data['TIMESTAMP'][trajectory[exit_id]]) - totime(data['TIMESTAMP'][trajectory[entry_id]]))
                time_diffs.append(time_diff) # change when expanding to multiple zones
                print(time_diff)

    transit_times = [x.total_seconds() for  x in time_diffs]
    transit_times = [x for x in transit_times if x>150] #just eliminating the early exitors
    # time analysis
    print("len(transit_times) ", len(transit_times))
    print("max(transit_times) ", max(transit_times))
    print("min(transit_times) ", min(transit_times))
    print("avg(transit_times) ", sum(transit_times)*1.0/len(transit_times))


    # speed analysis
    print("len(speed) ", len(speeds))
    print("max(speed) ", max(speeds))
    print("min(speed) ", min(speeds))
    print("avg(speed) ", sum(speeds)*1.0/len(speeds))

    plt.hist(transit_times, bins=[x for  x in range(0,int(max(transit_times))+15,15)])
    plt.show()
