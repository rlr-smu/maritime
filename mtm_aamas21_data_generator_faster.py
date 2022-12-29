'''
Generate the data for the AAMAS paper ShipGAN
'''
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd

from pandarallel import pandarallel
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from shapely.geometry import Point, Polygon
from matplotlib import pyplot as plt
import vaex
import math
from collections import defaultdict
import pickle

def getargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--vessel_data_file",  default='/DATA/chaithanya/mtm_24_months/SMU_Positions_2017B.csv', help="raw vessel data from marinetraffic") 
    parser.add_argument("--zone_data_file",  default='/home/chaithanya/work/maritime/encdata/jamesZones/zones.shp', help="enc data file showing tss zones") 
    parser.add_argument('--load_file', default= False, action='store_true', help='Load from datafile instead of cache')
    parser.add_argument('--cache_file', default= 'vessel_data.hdf5', help='file to store cached vessel data')
    parser.add_argument('--zoneid', type=int, default= 1, help='#')
    parser.add_argument('--subsetsize', type=int, default= 100000, help='When you only want work on a subset of data, for faster development')

    args = parser.parse_args()
    return args

def read_vessel_data(args, vessel_data_files):
    # read from file(read a subset for now)
    if args.load_file == True:
        
        dfs = []
        for fName in vessel_data_files:
            logger.info(fName)
            df = pd.read_csv(fName, sep=";", parse_dates=["TIMESTAMP_UTC"])
            dfs.append(df)
        df = vaex.from_pandas(pd.concat(dfs))
        if len(df) == 0:
            print("No samples in the given time range")
            exit()
        df = df.sort(["MMSI", "TIMESTAMP_UTC"])

        logger.info("Saving to cache")
        df.export(args.cache_file)
        logger.info("Saved to cache")
    else:
        logger.info("Using cached hdf file, instead of input files.")

    df = vaex.open(args.cache_file)

    return df


def read_zone_data(zone_data_file):
    # read from enc file
    gdf = gpd.GeoDataFrame.from_file(zone_data_file) # TODO: accommodate multiple zone files. currently only first zone
    gdf.plot()
    polygons = gdf.geometry.values
    for pind in range(len(polygons)):
        plt.text(polygons[pind].centroid.x, polygons[pind].centroid.y, str(pind),color='w')
    plt.show()
    print('len(polygons) ', len(polygons))
    return polygons

def add_arrow(line):

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    for i in range(0, len(xdata) - 1 ):
        line.axes.annotate('',
            xytext=(xdata[i], ydata[i]),
            xy=(xdata[i+1], ydata[i+1]),
            arrowprops=dict(arrowstyle="->"),
        )

def dist(x, y):
    return math.sqrt(x*x + y*y)

def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    All args must be of equal length.    
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

def getTransitTimes(args, vessel_data, zone_data):

    zone_length = {0:4, 1:4, 2:6, 3:7, 4:11
	,5:8, 6:3, 7:4, 8:4, 9:3
	,10:11, 11:7, 12:6.5, 13:6, 14:7
	, 15:6.5
    ,49:2, 50:2}
     # zone length in kms

    # print(zoneid)
    print(len(vessel_data))
    # print(zone_data[zoneid])
    
    min_chunk_size =  np.timedelta64(20, "m")
    chunk_separation_size = np.timedelta64(10, "m")
    long_jump_separation_size = 5 #km

     
    df = vessel_data
    logger.info("Calculating chunk separation points")
    df_shifted = df.shift(1)
    different_ship = df.MMSI.values != df_shifted.MMSI.values
    long_jumps = haversine_np(df.LON.values, df.LAT.values, df_shifted.LON.values, df_shifted.LAT.values) > long_jump_separation_size
    
    chunk_completes = (((df.TIMESTAMP_UTC.values - df_shifted.TIMESTAMP_UTC.values)) > chunk_separation_size) | different_ship | long_jumps
    df['x'] = df.LON
    df['y'] = df.LAT
    

    chunks = []
    last_index = 0
    time_stamps, speeds, x, y = df.TIMESTAMP_UTC.values,df.SPEED_KNOTSX10.values, df.x.values, df.y.values
    
    for i, is_chunk_complete  in enumerate(chunk_completes):
        if i%1000 == 0:
            print(f"Separating chunks progress: {100*i/len(chunk_completes):0.2f}%, {len(chunks)}", end="\r")
        if is_chunk_complete:
            if i - last_index > 10:
                start, end = df.TIMESTAMP_UTC.values[last_index], df.TIMESTAMP_UTC.values[i-1]
                if end-start > min_chunk_size:
                    chunks.append((last_index, i-1, start, end))   
            last_index = i+1

    currentzoneid = 11 # default. but is changed globally.
    def get_within_chunk(row):
        chunk = row.df
        TIMESTAMP_UTC, SPEED_KNOTSX10, x_original, y_original = chunk 

        # print(lat)
        points = [ Point(lon, lat) for lon, lat in zip(x_original, y_original)]
        is_within = gpd.GeoDataFrame(geometry=points).within(zone_data[currentzoneid]) # innermost operation

        #check neighbours, only for intersection zones
        neighs = [10,5,  51,52, 16,17,18]
        neighs = [11,14, 53,52, 16,17,18] # of zone 50
        # neighs = [5,16]
        

        in_zone = False
        durations = defaultdict(list)
        for i in range(len(is_within)-1):
            if not is_within[i] and is_within[i+1]:
                in_zone = True
                arrival_time = TIMESTAMP_UTC[i]
                arrival_step = i
                
            if is_within[i] and not is_within[i+1]:
                if in_zone:
                    departure_time = TIMESTAMP_UTC[i]
                    length_traversed_in_km = 111 * dist(x_original[i] - x_original[arrival_step]
                        ,y_original[i] - y_original[arrival_step])
                    
                    if length_traversed_in_km > 0.8 * zone_length[zoneid]:
                        arrzone = -1
                        depzone = -1
                        for neigh in neighs:
                            withinn_arr = gpd.GeoDataFrame(geometry=points[arrival_step:arrival_step+1]).within(zone_data[neigh])
                            if withinn_arr[0]:
                                arrzone = neigh
                            withinn_dep = gpd.GeoDataFrame(geometry=points[i+1:i+1+1]).within(zone_data[neigh])
                            if withinn_dep[0]:
                                depzone = neigh
                        if arrzone != -1 and depzone !=-1:
                            durations[arrzone*100+depzone].append((departure_time - arrival_time, SPEED_KNOTSX10[i]))
                in_zone = False
        return durations
    
    # within_zoneids = [i for i in range(13)]+ [i for i in range(14,20)]
    # within_zoneids = [16,17,18,49,50]
    within_zoneids = [50]
    def get_traffic_chunk(row):
        chunk = row.df
        TIMESTAMP_UTC, SPEED_KNOTSX10, x_original, y_original = chunk 

        step_size = np.timedelta64(1, 'm')
        t_original = (TIMESTAMP_UTC - TIMESTAMP_UTC.min())/ step_size
        minute_difference = (TIMESTAMP_UTC.max() - TIMESTAMP_UTC.min())/ step_size # float representing the steps
        minute_ids = np.arange(0, minute_difference, 1)
        
        # return 0
        x = np.interp(minute_ids, t_original, x_original)
        y = np.interp(minute_ids, t_original, y_original)

        # print(lat)
        # print('befor interp', len(x_original), 'after interp', len(x), TIMESTAMP_UTC.min(), TIMESTAMP_UTC.max())
        zone_ship_found = defaultdict(list)
        points = [ Point(lon, lat) for lon, lat in zip(x, y)]
        for zoneid in within_zoneids:
            is_within = gpd.GeoDataFrame(geometry=points).within(zone_data[zoneid]) # innermost operation

            ship_found_at = []
            for i in range(len(is_within)-1):
                if is_within[i]:
                    ship_found_at.append(TIMESTAMP_UTC.min() + minute_ids[i]*step_size)
            zone_ship_found[zoneid].append(ship_found_at)
        return zone_ship_found
    

    def to_chunk_values(row):
        s, e = row.start_index, row.end_index
        return (time_stamps[s:e+1], speeds[s:e+1], x[s:e+1], y[s:e+1])
    
    logger.info("calculating other params")
    
    pandarallel.initialize(progress_bar=True)
    chunks = pd.DataFrame(chunks, columns=['start_index','end_index', 'start_time', 'end_time'])
    chunks['df'] = chunks.apply(to_chunk_values, axis=1)

    def get_within(row):
        return get_within_chunk(row)

    #get transit times
    transittimesCalcuation = True
    if(transittimesCalcuation):
        for zoneid in within_zoneids:
            currentzoneid = zoneid
            print('running within function for zone ', currentzoneid)
            within_data = chunks.parallel_apply(get_within, axis=1)
            print('within data', type(within_data))
            print(len(within_data))
            # print(within_data)
            
            slistd = defaultdict(list)
            for l in within_data:
                for key, value in l.items():
                    slistd[key].extend(value)
            
            for zoneidd, slist in slistd.items():
                print('len slist: ', len(slist))
                transittimes_in_sec = [(t[0]/np.timedelta64(1, 's'), t[1]) for t in slist]

                print('len(transittimes)', len(transittimes_in_sec))
                print('transittimes[:10] ', transittimes_in_sec[:10])

                datasize = len(transittimes_in_sec)
                # transittimes_in_sec= sorted(transittimes_in_sec)[(datasize*10)//100:(datasize*90)//100] # taking only the 10 and 90 percentile.

                exportToFile(zoneidd, transittimes_in_sec)
        return
    
    #get traffic data TODO: need to interpolate 
    zone_traffic_export = {}
    for zoneid in within_zoneids:
        zone_traffic_export[zoneid] = defaultdict(int)

    zone_ship_founds = chunks.parallel_apply(get_traffic_chunk, axis=1)
    print('within data', type(zone_ship_founds))
    print("len(traffic_data)", len(zone_ship_founds))
    print(zone_ship_founds)
    for zone_ship_found in zone_ship_founds:
        for zone_id, traffic_data in zone_ship_found.items():
            for ship_found_at in traffic_data:
                for pt in np.array(ship_found_at, dtype='datetime64[m]'):
                    # print(pt)
                    zone_traffic_export[zone_id][pt] += 1 # this may not be adding things up by minute..

            
    def plot_daily_traffic(traffic_data):
        daily_traffic = defaultdict(list)
        for traffic in traffic_data:
            daily_traffic[pd.Timestamp(traffic).hour * 60 + pd.Timestamp(traffic).minute].append(traffic_data[traffic])
        
        xs = []
        avgs = []
        lens = []
        medians = []
        perc_25 = []
        perc_75 = []
        for traffic_minute,traffic_counts in sorted(daily_traffic.items()):
            lenn = len(traffic_counts)
            if(lenn > 0 ):
                xs.append(int(traffic_minute))
                lens.append(lenn)
                traffic_counts = sorted(traffic_counts)
                medians.append(traffic_counts[lenn//2])
                perc_25.append(traffic_counts[lenn*3//4])
                perc_75.append(traffic_counts[lenn//4])
                avgs.append(float(sum(traffic_counts)/ len(traffic_counts)))
                # print("len(traffic_counts)", len(traffic_counts))
        print(xs)
        print(avgs)
        # plt.plot([8],[0],'b.')
        # plt.axis('scaled')
        plt.plot(xs, avgs, label='avg traffic in zone'+str(currentzoneid))
        plt.plot(xs, lens, label='lens '+str(currentzoneid))
        plt.plot(xs, medians, label='medians '+str(currentzoneid))
        plt.plot(xs, perc_25, label='perc25 '+str(currentzoneid))
        plt.plot(xs, perc_75, label='perc75 ' +str(currentzoneid))
        plt.legend()
        plt.show()
    def save_daily_traffic(traffic_export, fname):
        with open(fname, 'wb') as f:
            pickle.dump(traffic_export, f)

    for zone_id in within_zoneids:
        save_daily_traffic(zone_traffic_export[zone_id], 'bulktraffic_data_'+str(zone_id)+'_'+str(len(zone_ship_founds))+'.pkl')
    # plot_daily_traffic(traffic_export)
    # plt.hist([t[0] for t in transittimes_in_sec], bins=[i*60 for i in range(240)])
    # plt.xlabel('time to cross zone (seconds)')
    # plt.ylabel('frequency')
    # plt.show()

    # plt.scatter([t[1] for t in transittimes_in_sec], [t[0] for t in transittimes_in_sec])
    # plt.xlabel('speed in x10 knots')
    # plt.ylabel('time to cross the zone(seconds)')
    # plt.show()

    


def exportToFile(zoneid, transittimes):
    print('exporting transittimes')
    with open('transittimes_2yr'+str(zoneid)+'.txt', 'w') as f:
        f.write(str(transittimes))

def main():
    args = getargs()

    vessel_data_files = ['2017A']
    vessel_data_files = ['/home/chaithanya/work/maritime/raw_data/SMU_Positions_' + v +'.csv' for v in vessel_data_files ]

    vessel_data = read_vessel_data(args, vessel_data_files)
    zone_data = read_zone_data(args.zone_data_file)


    getTransitTimes(args, vessel_data, zone_data)

    
    

if __name__ == '__main__':
    main()