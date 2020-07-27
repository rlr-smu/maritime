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
from collections import defaultdict
import random
'''
Most code for Maritime analysis is spaghetti, need to make the functionality more modular, as it is hampering my productivity,
Let's start from here..
'''

''' Parse Arguments '''

parser = argparse.ArgumentParser()
parser.add_argument("--ships_to_read",  type=int,  default=12, help="number of ships to check") 
parser.add_argument("--zone",  type=int,  default=11, help="zone for which you want the traffic_crossing time distribution") 
parser.add_argument("--save_traj_table",  type=bool,  default=False, help="Save the file back after interpolation") 
parser.add_argument("--zone_file",  default='/mnt/c/Users/chetu/work/maritime/encdata/jamesZones/zones.shp', help="Single zone file to track traffic") 
parser.add_argument("--workdir",  default='/mnt/c/Users/chetu/work/maritime/', help="working directory") 
parser.add_argument("--trajfile",  default='data/all_mmsi_traj.csv', help="trajectory meta file") 
parser.add_argument("--eventdir",  default='data/all_events/', help="trajectory meta file") 
parser.add_argument("--originalcsv",  default='data/two_ships.csv', help="non interpolated original file") 
parser.add_argument("--snapshot_timestep",  default='2019-07-06 19:22:00', help="time at which you need a snapshot of strait") 
parser.add_argument("--traffic_export_file_name",  default='/data/traffic_by_zone.csv', help="self explanatory") 



opt = parser.parse_args()
print(opt)
timestamp_format = '%Y-%m-%d %H:%M:%S'
startime = time.time()

tochinazones = [0, 1, 2, 3, 4, 5, 16, 17, 11, 12, 13]
toeuropezones = [19, 15, 14, 18, 10, 9, 8, 7, 6]


# shifted zones stitching
orig_to_rightshift_map = [18,17,16,10,9,8,15,14,13,12,11, 19,0,1,4,3,7,6,5,2]
orig_to_leftshift_map  = [15,14,13,12,11,10,19,18,17,16,9, 3,2,1,0,5,8,7,6,4]
orig_to_rightshift50_map = orig_to_rightshift_map
orig_to_leftshift50_map  = orig_to_leftshift_map

rightshift_zone_file_name = 'encdata/chetuZones/shifted_right25.shp'
leftshift_zone_file_name = 'encdata/chetuZones/shifted_left_25.shp'
rightshift50_zone_file_name = 'encdata/chetuZones/shifted_right50.shp'
leftshift50_zone_file_name = 'encdata/chetuZones/shifted_left_50.shp'

rightshift_traffic_file_name = 'data/rightshift25_traffic.csv'
leftshift_traffic_file_name  = 'data/leftshift25_traffic.csv'
rightshift50_traffic_file_name = 'data/rightshift50_traffic.csv'
leftshift50_traffic_file_name  = 'data/leftshift50_traffic.csv'


''' Define Functions '''

def totime(timestring):
    return datetime.strptime(timestring, timestamp_format)


def zone_overlap(zone_files1, zone_files2):
    '''
    Maps the overlap between the polygons defined by zone_files1 array and zonefiles2 array
    '''
    fig, ax = plt.subplots(figsize=(16,9))
    ax.set_xlim(103.535, 104.03)
    ax.set_ylim(1.02, 1.32)
    borderCol = 'black'
    collist = ['red', 'green', 'blue']
    for zoneFile in zone_files2:
        gdf = gpd.GeoDataFrame.from_file(zoneFile)
        gdf.plot(ax=ax, color='green', edgecolors=borderCol, alpha=0.5)
    for i,zoneFile in enumerate(zone_files1):
        gdf = gpd.GeoDataFrame.from_file(zoneFile)
        gdf.plot(ax=ax, color=collist[i%3], edgecolors='white', alpha=0.5)
    
    plt.show()

def zone_manipulation(zonefiles, zones_to_ignore= [3,4,5,6,7,8]):
    fig, ax = plt.subplots(figsize=(16,9))
    ax.set_xlim(103.535, 104.03)
    ax.set_ylim(1.02, 1.32)
    polylen = 65
    borderCol = ['black'] * polylen
    collist = ['green'] * polylen
    for i in zones_to_ignore:
        collist[i] = 'white'
        borderCol[i] = 'white'
    for zoneFile in zonefiles:
        gdf = gpd.GeoDataFrame.from_file(zoneFile)
        print(len(gdf.geometry.values))
        gdf.plot(ax=ax, color=collist, edgecolors=borderCol, alpha=0.5)

        polys = gdf.geometry.values
        for i in zones_to_ignore:
            print(i, polys[i])
        # New zones here  Custom Zones
        custom_polygons =[ [(103.659, 1.026),(103.650, 1.049),(103.648, 1.060),(103.691, 1.082),(103.700, 1.061)]
            , [(103.691, 1.082),(103.700, 1.061),(103.727, 1.083),(103.766, 1.127),(103.751, 1.137),(103.721, 1.098)]
            , [(103.766, 1.127),(103.751, 1.137),(103.754, 1.142),(103.796, 1.170),(103.820, 1.180),(103.826, 1.165),(103.810, 1.158),(103.769, 1.129),(103.766, 1.127)]
        ]

        for polygon_points in custom_polygons:
            longs = [x[0] for x in polygon_points]
            lats = [x[1] for x in polygon_points]
            print(polygon_points)
            poly = Polygon(zip(longs, lats))
            custom_polygon = gpd.GeoDataFrame(index=[0], crs=None, geometry=[poly])
            custom_polygon.plot(ax=ax, color='darkslategray', edgecolors=borderCol)

    plt.show()

def dist(p1, p2):
    '''l2 distance'''
    return math.sqrt( (p1[0]-p2[0]) **2 + (p1[1] -p2[1]) ** 2 )

def zone_extension(zonefiles):
    edit_zone=14
    fig, ax = plt.subplots(figsize=(16,9))
    ax.set_xlim(103.535, 104.03)
    ax.set_ylim(1.02, 1.32)

    gdf = gpd.GeoDataFrame.from_file(zonefiles[0])
    pols = gdf.geometry.values
    polcount = len(pols)
    print(polcount)
    cols = ['red'] * polcount
    borderCol = ['black'] * polcount

    cols[edit_zone] = 'gray'
    borderCol[edit_zone] = 'white'
    gdf.plot(ax=ax, color=cols, edgecolors=borderCol, alpha=0.5)
    print(pols[edit_zone])

    #extend polygon edit_zone in southern direction.

    #find the farthest edges from center.
    polygon = pols[edit_zone]
    centroid = polygon.centroid.coords[0]
    pol_pnts = polygon.boundary.coords
    
    vertexcount = len(pol_pnts)
    print("vertexcount ", vertexcount)
    edg_mpts = []
    for i in range(vertexcount):
        midpoint = ((pol_pnts[i][0] + pol_pnts[(i+1)% vertexcount][0]) / 2, (pol_pnts[i][1] + pol_pnts[(i+1)% vertexcount][1]) / 2 )
        distance_to_centroid = dist(midpoint, centroid)
        edg_mpts.append((distance_to_centroid, midpoint, i))
        plt.text(pol_pnts[i][0], pol_pnts[i][1], str(i))
    edg_mpts = sorted(edg_mpts)
    plt.show()

    rightid = edg_mpts[-1][2] 
    leftid  = edg_mpts[-2][2]
    # leftid = (edg_mpts[-2][2] +1 )%vertexcount

    #we'll assume that -1 to -2 is south (this will be wrong 50% of times.. TODO:fix it)
    
    dlon, dlat = 0.00, -0.0 # TODO: calculate this
    ext_pnts = [None] * (vertexcount-1)
    for i in range(vertexcount-1):
        ext_pnts[i] = (pol_pnts[i][0], pol_pnts[i][1])
    print(leftid, rightid)
    i=leftid
    while i != rightid:
        ext_pnts[i] = (pol_pnts[i][0] + dlon, pol_pnts[i][1] +  dlat)
        i = (i+vertexcount-2) %(vertexcount-1)
    
    longs = [x[0] for x in ext_pnts]
    lats = [x[1] for x in ext_pnts]
    plt.plot(longs, lats, 'r-')
    plt.plot(longs, lats, 'g.')
    print("ext_pnts")
    print(ext_pnts)
    poly = Polygon(zip(longs, lats))
    extended_polygon = gpd.GeoDataFrame(index=[0], crs=None, geometry=[poly])
    # extended_polygon.plot(ax=ax, color='darkslategray', edgecolors=borderCol, alpha=0.4)
    plt.show()


def polygon_feature(shp_file):
    fig, ax = plt.subplots(figsize=(16,9))
    ax.set_xlim(103.535, 104.03)
    ax.set_ylim(1.02, 1.32)
    borderCol = 'black'

    gdf = gpd.GeoDataFrame.from_file(shp_file)
    pols = gdf.geometry.values
    polcount = len(pols)
    print(polcount)
    cols = ['red'] * polcount
    cols[14] = 'green'
    gdf.plot(ax=ax, color=cols, edgecolors=borderCol, alpha=0.5)
    print(pols[14])

    plt.show()
    

def zone_crossing_dist(zonefiles, zoneid, shipmetafile, eventdir, allowed_traj_type):
    '''
    Get the distribution for zone crossing time, WITH interpolated data
    '''

    gdf = gpd.GeoDataFrame.from_file(zonefiles[0]) # TODO: accommodate multiple zone files. currently only first zone
    polygons = gdf.geometry.values
    poly_count = len(polygons)

    metadata = pd.read_csv(shipmetafile)
    metadatalen = min(opt.ships_to_read, len(metadata['MMSI_TRAJID']))
    metadata = metadata.replace({np.nan:None})

    trajectories = []
    for i in range(metadatalen):
        trajectories.append(pd.read_csv(eventdir+metadata['MMSI_TRAJID'][i]+'.csv'))
    print("len(trajectories): , ", len(trajectories),'\n\n')

    # more specific code here.

    interval = 1 # TODO: get this from metafile
    zones_to_plot = [49,50]
    
    time_diffs = {}
    speeds = []
    for pind in zones_to_plot:
        time_diffs[pind] = []
    for i, trajectory in enumerate(trajectories):
        traj_type = int(metadata['TRAJ_TYPE'][i])
        # print("traj_type: ", traj_type)
        if(traj_type not in allowed_traj_type): # only looking at 'to china' ships
            continue
        # print("no continue here")
        points = []
        traj_len = len(trajectory)
        for j in range(traj_len):
            latitude = trajectory['LAT'][j]
            longitude = trajectory['LON'][j]
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
            if(entry_id!=-1 and not already_inside):
                # speeds.append(data['SPEED (KNOTSx10)'][trajectory[exit_id]]) # appending both entry and exit speeds
                # speeds.append(data['SPEED (KNOTSx10)'][trajectory[entry_id]])
                
                time_diff = (totime(trajectory['TIMESTAMP'][exit_id]) - totime(trajectory['TIMESTAMP'][entry_id]))
                time_diffs[pind].append(time_diff) # change when expanding to multiple zones
                
                print(metadata['MMSI_TRAJID'][i], totime(trajectory['TIMESTAMP'][entry_id]), totime(trajectory['TIMESTAMP'][exit_id]))
                print(points[entry_id], points[exit_id])
                print(time_diff,'\n')
                
    for pind in zones_to_plot:
        transit_times = [x.total_seconds() for  x in time_diffs[pind]]
        # time analysis
        # print("TRANSIT TIMES")
        with open('transittimes_new_zone.txt', 'w') as f:
            f.write(str(transit_times))
        print("len(transit_times) ", len(transit_times))
        if(len(transit_times)>0):
            print("max(transit_times) ", max(transit_times))
            print("min(transit_times) ", min(transit_times))
            print("avg(transit_times) ", sum(transit_times)*1.0/len(transit_times))

            # plt.hist(transit_times, bins=[x for  x in range(0,int(max(transit_times))+15,15)])
            bin_size = 60
            xlen = int(max(transit_times))//bin_size + 1
            xs = [x*bin_size for x in range(xlen)]
            tcounts = [0] * xlen
            for x in transit_times:
                tcounts[int(x)//bin_size] +=1
            plt.plot(xs, tcounts, label=pind)
    plt.legend(bbox_to_anchor=(1.0, 1.1), loc='upper left', title="zone_id", borderaxespad=0.)
    plt.show()
    # # speed analysis
    # print("len(speed) ", len(speeds))
    # print("max(speed) ", max(speeds))
    # print("min(speed) ", min(speeds))
    # print("avg(speed) ", sum(speeds)*1.0/len(speeds))

def export_traffic_in_zone(zonefiles, shipmetafile, eventdir):
    '''Exports the traffic data (i.e a csv with time from july 1 to 28 and shows the traffic for each zone.)'''
    gdf = gpd.GeoDataFrame.from_file(zonefiles[0]) # TODO: accommodate multiple zone files. currently only first zone
    polygons = gdf.geometry.values
    poly_count = len(polygons)

    metadata = pd.read_csv(shipmetafile)
    metadatalen = min(opt.ships_to_read, len(metadata['MMSI_TRAJID']))
    metadata = metadata.replace({np.nan:None})

    trajectories = []
    vessel_counts = {}
    zones = toeuropezones + tochinazones
    for pind in zones:
        vessel_counts[pind]=defaultdict(int)
    for i in range(metadatalen):
        trajectories.append(pd.read_csv(eventdir+metadata['MMSI_TRAJID'][i]+'.csv'))
    print("len(trajectories): , ", len(trajectories),'\n\n')
    max_vessel_count = 0
    for i, trajectory in enumerate(trajectories):
        if(i%100 ==0):
            print("trajectory index ", i, "time: %.2f" % (time.time()-startime))
        traj_type = int(metadata['TRAJ_TYPE'][i])
        # print("traj_type: ", traj_type)
        if(traj_type not in allowed_traj_type): # only looking at 'to china' ships
            continue
        # print("no continue here")
        points = []
        traj_len = len(trajectory)
        for j in range(traj_len):
            latitude = trajectory['LAT'][j]
            longitude = trajectory['LON'][j]
            points.append(Point(longitude, latitude))
        shp_pnts = gpd.GeoDataFrame(geometry=points)
        
        
        for pind in zones:
            vals = shp_pnts.within(polygons[pind])
            for pnt_key, is_inside in enumerate(vals):
                if(is_inside):
                    # print("inside ", pind, "at", totime(trajectory['TIMESTAMP'][pnt_key]))
                    vessel_counts[pind][totime(trajectory['TIMESTAMP'][pnt_key])] += 1
                    vessel_count = vessel_counts[pind][totime(trajectory['TIMESTAMP'][pnt_key])]
                    if(max_vessel_count < vessel_count):
                        max_vessel_count = vessel_count
                        print("max_vessel_count", max_vessel_count, pind, trajectory['TIMESTAMP'][pnt_key])
        
    print("max_vessel_count : ", max_vessel_count)

    ### NOW EXPORT THE DATA TO A FILE
    traffic_dataframe_file = opt.workdir + opt.traffic_export_file_name
    traffic_df = pd.DataFrame(columns=['TIMESTEP']+zones)
    first_timestep =  totime('2019-07-01 00:01:00')
    last_timestep =  totime('2019-07-28 23:59:00')
    timestep = first_timestep
    i=1
    while timestep < last_timestep:
        if(i%1440 ==0):
            print("day ", i/1440)
        zonetraffics = []
        for zone in toeuropezones+tochinazones:
            zonetraffics.append(vessel_counts[zone][timestep])
        traffic_df.loc[i]= [str(timestep)]+zonetraffics

        i += 1
        timestep = timestep + timedelta(minutes=1)
    # for i, key in enumerate(timesteps):
        
    # for i, key in enumerate(trajectories):
    #     traffic_df.loc[i]= [key] + [ data[x][trajectories[key][0]] for x in columns[1:6] ] + [traj_type[key]]
    
    traffic_df.to_csv(traffic_dataframe_file, index=False)
    


def traffic_in_zone(zoneid, timestep, traffic_dataframe):
    initial_time = totime('2019-07-01 00:01:00')
    # print(timestep)
    rowid = int((totime(timestep) - initial_time).total_seconds()/60)
    # print(zoneid, rowid)
    # print(traffic_dataframe)
    traffic = 15
    if(rowid < 40300):
        traffic = traffic_dataframe[str(zoneid)][rowid]

    return traffic

def zone_crossing_time_grouped_by_traffic(zonefiles, zoneid, shipmetafile, eventdir, allowed_traj_type):
    '''
    Get the distribution for zone crossing time, WITH interpolated data
    '''
    print("zone_crossing_time_grouped_by_traffic")
    gdf = gpd.GeoDataFrame.from_file(zonefiles[0]) # TODO: accommodate multiple zone files. currently only first zone
    polygons = gdf.geometry.values
    poly_count = len(polygons)

    # right and left shifted polygons
    rightpolygons = gpd.GeoDataFrame.from_file(rightshift_zone_file_name).geometry.values
    leftpolygons = gpd.GeoDataFrame.from_file(leftshift_zone_file_name).geometry.values
    right50polygons = gpd.GeoDataFrame.from_file(rightshift50_zone_file_name).geometry.values
    left50polygons = gpd.GeoDataFrame.from_file(leftshift50_zone_file_name).geometry.values

    metadata = pd.read_csv(shipmetafile)
    metadatalen = min(opt.ships_to_read, len(metadata['MMSI_TRAJID']))
    metadata = metadata.replace({np.nan:None})

    traffic_dataframe_files = [opt.traffic_export_file_name, rightshift_traffic_file_name, leftshift_traffic_file_name, rightshift50_traffic_file_name, leftshift50_traffic_file_name]

    traffic_dataframes = [pd.read_csv(opt.workdir + f).replace({np.nan:None}) for f in traffic_dataframe_files]

    trajectories = []
    for i in range(metadatalen):
        trajectories.append(pd.read_csv(eventdir+metadata['MMSI_TRAJID'][i]+'.csv'))
    print("len(trajectories): , ", len(trajectories),'\n\n')

    fig, ax = plt.subplots(figsize=(16,9))
    ax.set_xlim(-1, 3600)
    logscale = False
    if(logscale):
        ax.set_yscale('log')
    # ax.set_ylim(1.02, 1.32)
    # more specific code here.

    interval = 1 # TODO: get this from metafile
    zones_to_plot = [zoneid]
    
    time_diffs = {}
    speeds = []

    leftshift_zoneid = orig_to_leftshift_map[zoneid]
    rightshift_zoneid = orig_to_rightshift_map[zoneid]



    for pind in zones_to_plot:
        time_diffs[pind] = []
        for traffic_count in range(15):
            time_diffs[pind].append([]) 
    for i, trajectory in enumerate(trajectories):
        if(i%100==0):
            print("trajectory index ", i, "time: %.2f" % (time.time()-startime))
        traj_type = int(metadata['TRAJ_TYPE'][i])
        # print("traj_type: ", traj_type)
        if(traj_type not in allowed_traj_type): # only looking at 'to china' ships
            continue
        # print("no continue here")
        points = []
        traj_len = len(trajectory)
        for j in range(traj_len):
            latitude = trajectory['LAT'][j]
            longitude = trajectory['LON'][j]
            points.append(Point(longitude, latitude))
        shp_pnts = gpd.GeoDataFrame(geometry=points)
                
        for pind in zones_to_plot:
            # print(polygon)
            
            vals = shp_pnts.within(polygons[pind])
            valsright = shp_pnts.within(rightpolygons[orig_to_rightshift_map[pind]])
            valsleft =  shp_pnts.within(leftpolygons[orig_to_leftshift_map[pind]])
            valsright50 = shp_pnts.within(right50polygons[orig_to_rightshift50_map[pind]])
            valsleft50 =  shp_pnts.within(left50polygons[orig_to_leftshift50_map[pind]])
            listofvals = [vals, valsright, valsleft, valsright50, valsleft50]

            def get_ship_entry_exit_times(vals):
                already_inside = False
                entry_id = -1
                exit_id = -1
                for pnt_key, is_inside in enumerate(vals):
                    if(is_inside):
                        if(not already_inside):
                            entry_id = pnt_key
                        already_inside = True
                        exit_id = pnt_key

                    elif(already_inside == True):
                        already_inside = False
                        break # TODO: no need to break if taken care of multiple entry to a zone in a trajectory
                return entry_id, exit_id, already_inside

            for i, vals in enumerate(listofvals):
                entry_id, exit_id, already_inside = get_ship_entry_exit_times(vals)

                if(entry_id!=-1 and not already_inside):
                    # speeds.append(data['SPEED (KNOTSx10)'][trajectory[exit_id]]) # appending both entry and exit speeds
                    # speeds.append(data['SPEED (KNOTSx10)'][trajectory[entry_id]])
                    
                    entry_time = trajectory['TIMESTAMP'][entry_id]
                    traffic = traffic_in_zone(pind, entry_time, traffic_dataframes[i]) # get this data from another file.
                    
                    time_diff = (totime(trajectory['TIMESTAMP'][exit_id]) - totime(trajectory['TIMESTAMP'][entry_id]))
                    if(traffic<15):
                        time_diffs[pind][traffic].append(time_diff) # change when expanding to multiple zones
                    
                    # print(metadata['MMSI_TRAJID'][i], totime(trajectory['TIMESTAMP'][entry_id]), totime(trajectory['TIMESTAMP'][exit_id]))
                    # print(points[entry_id], points[exit_id])
                    # print(time_diff,'\n')
                
    for i, traffic_counts in enumerate(time_diffs[zoneid]):
        
        transit_times = [x.total_seconds() for  x in traffic_counts]
        # time analysis
        # print("TRANSIT TIMES")
        # with open('transittimes.txt', 'w') as f:
        #     f.write(str(transit_times))
        print("\nlen(transit_times) ", len(transit_times))
        if(len(transit_times)>0):
            print("max(transit_times) ", max(transit_times))
            print("min(transit_times) ", min(transit_times))
            print("avg(transit_times) ", sum(transit_times)*1.0/len(transit_times))

            # plt.hist(transit_times, bins=[x for  x in range(0,int(max(transit_times))+15,15)])
            bin_size = 60
            xlen = int(max(transit_times))//bin_size + 1
            xs = [x*bin_size for x in range(xlen)]
            tcounts = [1 if logscale else 0] * xlen
            for x in transit_times:
                tcounts[int(x)//bin_size] +=1
            ax.plot(xs, tcounts, label=i)
    ax.legend(bbox_to_anchor=(1.0, 1.1), loc='upper left', title=str(zoneid) + "zone traffic_count", borderaxespad=0.)
    # plt.show()
    plt.savefig('shifted' + ('log_' if logscale else '')+'zone_' + str(zoneid)+'_grouped_by_traffic.png')




def shipcount_in_zones(zonefiles, zones, allowed_traj_type):
    gdf = gpd.GeoDataFrame.from_file(zonefiles[0]) # TODO: accommodate multiple zone files. currently only first zone
    polygons = gdf.geometry.values
    poly_count = len(polygons)

    metadata = pd.read_csv(shipmetafile)
    metadatalen = min(opt.ships_to_read, len(metadata['MMSI_TRAJID']))
    metadata = metadata.replace({np.nan:None})

    trajectories = []
    vessel_counts = {}
    for pind in zones:
        vessel_counts[pind]=defaultdict(int)
    for i in range(metadatalen):
        trajectories.append(pd.read_csv(eventdir+metadata['MMSI_TRAJID'][i]+'.csv'))
    print("len(trajectories): , ", len(trajectories),'\n\n')
    max_vessel_count = 0
    for i, trajectory in enumerate(trajectories):
        if(i%100 ==0):
            print("trajectory index ", i, "time: %.2f" % (time.time()-startime))
        traj_type = int(metadata['TRAJ_TYPE'][i])
        # print("traj_type: ", traj_type)
        if(traj_type not in allowed_traj_type): # only looking at 'to china' ships
            continue
        # print("no continue here")
        points = []
        traj_len = len(trajectory)
        for j in range(traj_len):
            latitude = trajectory['LAT'][j]
            longitude = trajectory['LON'][j]
            points.append(Point(longitude, latitude))
        shp_pnts = gpd.GeoDataFrame(geometry=points)
        
        
        for pind in zones:
            vals = shp_pnts.within(polygons[pind])
            for pnt_key, is_inside in enumerate(vals):
                if(is_inside):
                    vessel_counts[pind][totime(trajectory['TIMESTAMP'][pnt_key])] += 1
                    vessel_count = vessel_counts[pind][totime(trajectory['TIMESTAMP'][pnt_key])]
                    if(max_vessel_count < vessel_count):
                        max_vessel_count = vessel_count
        
    
    for pind, timedist in vessel_counts.items():
        num_timesteps = defaultdict(int)
        for timestamp, vessel_count in timedist.items():
            if(vessel_count >= max_vessel_count):
                print("high traffic at zone",pind, timestamp, vessel_count)
            num_timesteps[vessel_count] +=1
        print("\n", pind)
        print(dict(num_timesteps))



def getzoneid(lon, lat, polygons):
    '''Returns the polygon id in the polygons where lat lon is found.
    If not found in any zone, returns -1
    '''
    points = [Point(lon,lat)]
    shp_pnts = gpd.GeoDataFrame(geometry=points)
    polylen = len(polygons)

    for pind in range(polylen):
        vals = shp_pnts.within(polygons[pind])
        for pnt_key, is_inside in enumerate(vals):
            if(is_inside):
                return pind
    return -1
    

def compress(arr):
    ''' compresses equal values eg: [1,1,5,5,5,6,6,7,9,9,9] becomes [1,5,6,7,9]
    '''
    ans = []
    if(len(arr)>0):
        ans.append(arr[0])
    for i in range(1,len(arr)):
        if(arr[i] != arr[i-1]):
            ans.append(arr[i])
    return ans

        

    
def interpolation_validation(zonefiles, shipmetafile, eventdir, originalcsv):
    '''
    validate if the interpolation has happened properly
    '''
    gdf = gpd.GeoDataFrame.from_file(zonefiles[0]) # TODO: accommodate multiple zone files. currently only first zone
    polygons = gdf.geometry.values
    poly_count = len(polygons)

    metadata = pd.read_csv(shipmetafile)
    metadatalen = min(opt.ships_to_read, len(metadata['MMSI_TRAJID']))
    metadata = metadata.replace({np.nan:None})

    int_trajectories = []
    for i in range(metadatalen):
        int_trajectories.append(pd.read_csv(eventdir+metadata['MMSI_TRAJID'][i]+'.csv'))
    print("len(int_trajectories): ", len(int_trajectories),'\n\n')

    orig_zones = {}
    # get non interpolated trajectories from originalcsv
    orig_csv = pd.read_csv(originalcsv)
    orig_len = len(orig_csv)

    for i in range(orig_len):
        if(i%100 ==0):
            print("orig_file line ", i, "/", orig_len, "time: %.2f" % (time.time()-startime))
        mmsi = orig_csv['MMSI'][i]
        if(mmsi in orig_zones):
            orig_zones[mmsi].append(getzoneid(orig_csv['LON'][i], orig_csv['LAT'][i], polygons))
        else:
            orig_zones[mmsi] = [getzoneid(orig_csv['LON'][i], orig_csv['LAT'][i], polygons)]
    
    #compare if the original matches with interpolated
    lendiff, zonediff, good_interpolation = 0,0,0
    timekeep = 0
    for mmsi, ord_zones in orig_zones.items():
        timekeep += 1
        if(timekeep % 1 ==0):
            print("comparison ", timekeep , "/~",metadatalen, ":::", lendiff, "+", zonediff, "+", good_interpolation, "=", lendiff + zonediff + good_interpolation
                ,"time: %.2f" % (time.time()-startime))
        trajid = 0
        int_zones = []
        int_key= str(mmsi)+'_'+str(trajid)

        while int_key in metadata['MMSI_TRAJID'].values:
            tindex = metadata['MMSI_TRAJID'][metadata['MMSI_TRAJID'] == int_key].index.values[0]
            # print("tindex", tindex)
            trajlen = len(int_trajectories[tindex])
            for i in range(trajlen):
                int_zones.append(getzoneid(int_trajectories[tindex]['LON'][i], int_trajectories[tindex]['LAT'][i], polygons))
            trajid += 1

            int_key =  str(mmsi)+'_'+str(trajid)
        
        #compare int_zones and ord_zones basically if uniq(int)= uniq(ord)
        print('\n', mmsi," ::")
        # print("int_zones: ",int_zones)
        # print("ord_zones: ",ord_zones)

        # remove -1s from zones.
        int_zones = [i for i in int_zones if i!=-1]
        ord_zones = [i for i in ord_zones if i!=-1]

        comp_int = compress(int_zones)
        comp_ord = compress(ord_zones)

        print("comp_int: ",comp_int)
        print("comp_ord: ",comp_ord)

        if(len(comp_int) != len(comp_ord)):
            lendiff +=1
        else:
            wrong =False
            for i,j in zip(comp_int, comp_ord):
                if(i!=j):
                    wrong = True
                    break
            if(wrong):
                zonediff += 1
            else:
                good_interpolation += 1
            
    
    print("lendiff trajs:", lendiff)
    print("zonediff trajs:", zonediff)
    print("good_interpolation trajs:", good_interpolation)


def export_to_shapefile():
    fig, ax = plt.subplots(figsize=(16,9))
    ax.set_xlim(103.535, 104.03)
    ax.set_ylim(1.02, 1.32)
    polylen = 65
    borderCol = ['black'] * polylen
    collist = ['green'] * polylen

    custom_polygons =[ 
    [(103.691, 1.082),(103.700, 1.061),(103.727, 1.083),(103.766, 1.127),(103.751, 1.137),(103.721, 1.098)]
    ]

    for polygon_points in custom_polygons:
        longs = [x[0] for x in polygon_points]
        lats = [x[1] for x in polygon_points]
        print(polygon_points)
        poly = Polygon(zip(longs, lats))
        custom_polygon = gpd.GeoDataFrame(index=[0], crs=None, geometry=[poly])
        custom_polygon.plot(ax=ax, color='darkslategray', edgecolors=borderCol)
        custom_polygon.to_file('merged56zone.shp')
    plt.show()



def show_map(zonefiles):
    fig, ax = plt.subplots(figsize=(16,9))
    ax.set_xlim(103.535, 104.05)
    ax.set_ylim(1.02, 1.32)

    gdf = gpd.GeoDataFrame.from_file(zonefiles[0]) # TODO: accommodate multiple zone files. currently only first zone
    polygons = gdf.geometry.values
    poly_count = len(polygons)

    gdf.plot(ax = ax, color = 'gray')
    for pind, polygon in enumerate(polygons):
        pind_text = plt.text(polygon.centroid.x, polygon.centroid.y, str(pind),color='w')

    metadata = pd.read_csv(shipmetafile)
    metadatalen = min(opt.ships_to_read, len(metadata['MMSI_TRAJID']))
    metadata = metadata.replace({np.nan:None})

    trajectories = []
    for i in range(metadatalen):
        trajectories.append(pd.read_csv(eventdir+metadata['MMSI_TRAJID'][i]+'.csv'))

    for i, trajectory in enumerate(trajectories):
        lats = []
        lons = []
        traj_len = len(trajectory)
        for j in range(traj_len):
            lats.append(trajectory['LAT'][j])
            lons.append(trajectory['LON'][j])
        lll = 100
        plt.plot(lons[:lll], lats[:lll], 'b-')
        plt.plot(lons[lll-1:], lats[lll-1:], 'r-')
        # plt.plot(lons, lats, 'g.')


    plt.show()

def traffic_map():
    # this whole module is a hack to compensate for inaccurate traffic data: TODO: fix traffic.json file.. 
    import json
    mapp ={}
    fig, ax = plt.subplots()

    with open('traffic.json', 'r') as jsonFile:
        json_dict = json.load(jsonFile)
        prev_zone_map = {}
        # with open('fixed_data.json', 'w') as f:
        #     json.dump(json_dict, f)

        print("len: ", len(json_dict))
        for zone in json_dict:
            for zone_id, mapp in zone.items():
                if(int(zone_id) in tochinazones):
                    xs = []
                    ys = []
                    pairs = []
                    for k, v in mapp.items():
                        prev_zone_map_k = 0
                        if k in prev_zone_map:
                            prev_zone_map_k = prev_zone_map[k]
                        pairs.append((int(k), mapp[k] - prev_zone_map_k))

                    # pairs = [(int(k),v) for k, v in mapp.items()]
                    for (x,y) in sorted(pairs):
                        xs.append(x)
                        ys.append(y)
                    ax.set_yscale('log')
                    ax.plot(xs, ys, label=zone_id)
                prev_zone_map = mapp
    plt.xlabel('(n)Number of ships in the zone')
    plt.ylabel('Number of occurences with n ships in the zone')
    plt.legend(bbox_to_anchor=(0.9, 1.1), loc='upper left', title="to_china_zones", borderaxespad=0.)
    plt.show()

def satellite_map_snapshot(zonefiles, timestep):
    ''' Gives the port snapshot at this timestep'''

    fig, ax = plt.subplots(figsize=(16,9))
    ax.set_xlim(103.535, 104.03)
    ax.set_ylim(1.02, 1.32)
    
    gdf = gpd.GeoDataFrame.from_file(zonefiles[0]) # TODO: accommodate multiple zone files. currently only first zone
    polygons = gdf.geometry.values
    poly_count = len(polygons)
    gdf.plot(ax=ax, color='gray', edgecolors='black', alpha=0.5)

    metadata = pd.read_csv(shipmetafile)
    metadatalen = min(opt.ships_to_read, len(metadata['MMSI_TRAJID']))
    metadata = metadata.replace({np.nan:None})

    trajectories = []
    for i in range(metadatalen):
        trajectories.append(pd.read_csv(eventdir+metadata['MMSI_TRAJID'][i]+'.csv'))
    print("len(trajectories): , ", len(trajectories),'\n\n')

    ship_lons =[]
    ship_lats =[]
    
    for i, trajectory in enumerate(trajectories):
        if(i%100 ==0):
            print("trajectory index ", i, "time: %.2f" % (time.time()-startime))
        
        traj_len = len(trajectory)
        if(traj_len ==0): # TODO: why is it zero? need to fix this
            continue
        # print("\ntrajectory: ", i, "len", traj_len,metadata['MMSI_TRAJID'][i] )
        traj_start = totime(trajectory['TIMESTAMP'][0])
        traj_end = totime( trajectory['TIMESTAMP'][traj_len-1])
        # print("traj start time:", traj_start)
        # print("traj end time:", traj_end)
        
        
        if(timestep >=traj_start and timestep <=traj_end):
            # print("this traj contributes to satellite map", i)
            idx = (timestep-traj_start).total_seconds()/60
            lat = trajectory['LAT'][idx]
            lon = trajectory['LON'][idx]
            ship_lons.append(lon)
            ship_lats.append(lat)

    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', title=str(timestep), borderaxespad=0.)
    plt.plot(ship_lons, ship_lats, 'g.')
    plt.show()

def plot_transit_times():

    fig, ax = plt.subplots(figsize=(16,9))
    ax.set_xlim(-1, 3600)
    ax.set_ylim(0, 400)

    files = ['transittimes.txt', 'transittimes_new_zone.txt']
    timeslist = []
    for i in range(2):
        with open(files[i], 'r') as f:
            timeslist.append(eval(f.read()))

    bin_size = 60
    for i in range(2):
        xlen = int(max(timeslist[i]))//bin_size + 1
        xs = [x*bin_size for x in range(xlen)]
        tcounts = [0] * xlen
        for x in timeslist[i]:
            tcounts[int(x)//bin_size] +=1
        ax.plot(xs, tcounts, label=i)
    plt.xlabel('time taken to cross zone(seconds)')
    plt.ylabel('Number of occurences')
    
    plt.legend(bbox_to_anchor=(1.0, 1.1), loc='upper left', title="shifted zones", borderaxespad=0.)
    plt.show()
    

''' Main '''

# check zone overlap
workdir = opt.workdir
zonefiles1 = [workdir+'/encdata/shp_sg5c4030/TSSLPT.shp'
    ,workdir+'/encdata/shp_sg5c4031/TSSLPT.shp'
    ,workdir+'/encdata/shp_sg5c4032/TSSLPT.shp'
    ,workdir+'/encdata/shp_sg5c4034/TSSLPT.shp'
    ,workdir+'/encdata/shp_sg5c4035/TSSLPT.shp'
    ,workdir+'/encdata/shp_sg5c4036/TSSLPT.shp'
    ,workdir+'/encdata/shp_sg5c4037/TSSLPT.shp'
    ,workdir+'/encdata/shp_sg5c4039/TSSLPT.shp'
    ,workdir+'/encdata/shp_sg5c4041/TSSLPT.shp'
    ,workdir+'/encdata/shp_sg5c4043/TSSLPT.shp']
zonefiles2 = [opt.zone_file]

shipmetafile  = workdir + opt.trajfile
eventdir = workdir + opt.eventdir
originalcsv = workdir + opt.originalcsv

''' FUNCTION CALLS'''
# zone_overlap(zonefiles1, zonefiles2)
# zone_manipulation(zonefiles2)
# polygon_feature(zonefiles2[0])
# zone_extension(zonefiles2)

# allowed_traj_type = [5,4,1] # for to china zones
allowed_traj_type = [2,3,7,5,4,1]
zone = 49

# shipcount_in_zones(zonefiles2, tochinazones + toeuropezones, allowed_traj_type)

timestep = totime('2019-07-01 05:34:00')
# timestep = totime('2019-07-15 15:23:00')
timestep = totime(opt.snapshot_timestep)

# satellite_map_snapshot(zonefiles2, timestep)
# show_map(zonefiles2)
# zone_crossing_dist(zonefiles2, zone, shipmetafile, eventdir, allowed_traj_type)

zone_crossing_time_grouped_by_traffic(zonefiles2, opt.zone, shipmetafile, eventdir, allowed_traj_type)

# export_traffic_in_zone(zonefiles2, shipmetafile, eventdir)

# interpolation_validation(zonefiles2, shipmetafile, eventdir, originalcsv)
# export_to_shapefile()

# traffic_map()

# plot_transit_times()