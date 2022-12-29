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
    for pind in range(20):
        plt.text(polygons[pind].centroid.x, polygons[pind].centroid.y, str(pind),color='w')
    # plt.show()
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

def getTransitTimes(args, zoneid, vessel_data, zone_data):

    zone_length = {0:4, 1:4, 2:6, 3:7, 4:11
	,5:8, 6:3, 7:4, 8:4, 9:3
	,10:11, 11:7, 12:6.5, 13:6, 14:7
	, 15:6.5} # zone length in kms

    print(zoneid)
    print(len(vessel_data))
    print(zone_data[zoneid])
    
    # plt.plot(zone_data[zoneid])
    # plt.show()
        
    def is_within(row):
        # print(row)
        lat, lon = row.LAT , row.LON
        # print(lat)
        return gpd.GeoDataFrame(geometry=[ Point(lon, lat) ]).within(zone_data[zoneid])[0]

    pandarallel.initialize(progress_bar=True)
    vessel_data_cp = vessel_data[:args.subsetsize].to_pandas_df()
    print(type(vessel_data))
    print(vessel_data)
    print(type(vessel_data_cp))
    print('len(vessel_data_cp)', len(vessel_data_cp))
    print('len(vessel_data)', len(vessel_data))
    in_polygon = vessel_data_cp.parallel_apply(is_within, axis=1)
    print(type(in_polygon))
    print(in_polygon)
    
    in_polygon = in_polygon
    in_polygon_1 = in_polygon.shift(1, fill_value=False)
    print(type(in_polygon_1))
    # print(in_polygon)
    # print(in_polygon_1)
    arrivals =  (in_polygon & ~in_polygon_1) | (~in_polygon & in_polygon_1) # XOR operation on shifted values.
    # departures =  (in_polygon == True &  in_polygon_1 == False)

    print(len(arrivals))
    # print(arrivals)

    arrived = False
    arrival_time = None
    arrival_step = 0
    timedurations= []
    for i, change in enumerate(arrivals):
        if change:
            if arrived:
                departure_time = vessel_data.TIMESTAMP_UTC.values[i]
                if vessel_data.MMSI.values[arrival_step] == vessel_data.MMSI.values[i]: # Need to make this parallel
                    length_traversed_in_km = 111 * dist(vessel_data.LON.values[i] - vessel_data.LON.values[arrival_step]
                        ,vessel_data.LAT.values[i] - vessel_data.LAT.values[arrival_step])
                    if length_traversed_in_km > 0.8 * zone_length[zoneid]: # record data only if the vessel has gone the length of zone
                        timedurations.append((departure_time - arrival_time, vessel_data.SPEED_KNOTSX10.values[i]))

                        # plot the path through the zone
                        if timedurations[-1][0]/np.timedelta64(1, 's') < 180: # less than x seconds 
                            extrapts = 5
                            shiplats = [vessel_data.LAT.values[ts] for ts in range(arrival_step-extrapts, i+extrapts)]
                            shiplons = [vessel_data.LON.values[ts] for ts in range(arrival_step-extrapts, i+extrapts)]
                            line = plt.plot(shiplons, shiplats,'o-')[0]
                            add_arrow(line)
                            
                # print( arrival_time, departure_time, vessel_data.MMSI.values[i], i)
                arrived = False
            else:
                arrived = True
                arrival_time = vessel_data.TIMESTAMP_UTC.values[i]
                arrival_step = i
                # print('arrival time', arrival_time, vessel_data.MMSI.values[i], i)
        if i > args.subsetsize:
            break

    plt.show()
    # print(len(depar   tures))
    print(len(timedurations))
    # print(timedurations)
    transittimes_in_sec = [(t[0]/np.timedelta64(1, 's'), t[1]) for t in timedurations]
    return transittimes_in_sec



def exportToFile(transittimes):
    print('exporting transittimes')
    with open('transittimes_2yr.txt', 'w') as f:
        f.write(str(transittimes))

def main():
    args = getargs()

    vessel_data_files = ['2017A', '2017B', '2018A', '2018B', '2019A']
    vessel_data_files = ['/DATA/chaithanya/mtm_24_months/SMU_Positions_' + v +'.csv' for v in vessel_data_files ]

    vessel_data = read_vessel_data(args, vessel_data_files)
    zone_data = read_zone_data(args.zone_data_file)


    transittimes_in_sec = getTransitTimes(args, args.zoneid, vessel_data, zone_data)

    print('len(transittimes)', len(transittimes_in_sec))
    print('transittimes[:10] ', transittimes_in_sec[:10])

    datasize = len(transittimes_in_sec)
    # transittimes_in_sec= sorted(transittimes_in_sec)[(datasize*10)//100:(datasize*90)//100] # taking only the 10 and 90 percentile.

    exportToFile(transittimes_in_sec)
    
    plt.hist([t[0] for t in transittimes_in_sec], bins=[i*60 for i in range(240)])
    plt.xlabel('time to cross zone (seconds)')
    plt.ylabel('frequency')
    plt.show()

    plt.scatter([t[1] for t in transittimes_in_sec], [t[0] for t in transittimes_in_sec])
    plt.xlabel('speed in x10 knots')
    plt.ylabel('time to cross the zone(seconds)')
    plt.show()

    

if __name__ == '__main__':
    main()