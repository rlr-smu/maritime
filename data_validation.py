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

parser = argparse.ArgumentParser()
parser.add_argument("--latlonfile",  default='data/two_ships.csv', help='file with ship location data(AIS)') 
parser.add_argument("--rows_to_read", type=int,  default=1000, help="limit the number of rows from latlonfile") 
parser.add_argument("--interval", type=int,  default=2, help="time interval for interpolation") 
parser.add_argument("--column_index", type=int,  default=6, help="0. MMSI,VESSEL TYPE,LENGTH,WIDTH,IMO,5.DESTINATION,STATUS,SPEED (KNOTSx10),LON,LAT,10.COURSE,HEADING,TIMESTAMP") 
parser.add_argument("--bucket_count", type=int,  default=100, help="total buckets.") 
parser.add_argument("--zone_file",  default='/mnt/c/Users/chetu/work/maritime/encdata/jamesZones/zones.shp', help="Single zone file to track traffic") 
parser.add_argument("--show_map",  type=bool, default=False ,help="Show colored zone map?") 
parser.add_argument("--lastnavg",  type=int,  default=10, help="avg of last n points in traffic plot.") 
parser.add_argument("--exit_entry_diff",  type=int,  default=12, help="difference between entry and exit of a ship from map") 
parser.add_argument("--zone_traffic",  type=bool,  default=False, help="display zone traffic") 
parser.add_argument("--separate_zone_plot",  type=bool,  default=False, help="plot each zone or sum of zones") 
parser.add_argument("--daily_bins",  type=bool,  default=False, help="daily bins?") 

opt = parser.parse_args()
print(opt)

vessel_csv_filename=opt.latlonfile
rows_to_read = opt.rows_to_read
bucket_count = opt.bucket_count
interval = opt.interval
timestamp_format = '%Y-%m-%d %H:%M:%S'

data = pd.read_csv(vessel_csv_filename)
datalen = min(rows_to_read, len(data["MMSI"]))
data = data.replace({np.nan:None})

columns = ['MMSI', 'VESSEL TYPE', 'LENGTH', 'WIDTH', 'IMO', 'DESTINATION', 'STATUS', 'SPEED (KNOTSx10)', 'LON', 'LAT', 'COURSE', 'HEADING', 'TIMESTAMP']
attribute = columns[opt.column_index]
###-------------------------------------------------------

timestamp_format = '%Y-%m-%d %H:%M:%S'
trajectories={}

maxlon = 104.03 # as per quote
minlon = 103.535 # as per quote
maxlon -= 0.05 # increase prob of trajectory breaking
minlon += 0.05 # increase prob of trajectory breaking


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
	r = 0
	# for i in range(len(arr)):
	# 	if(arr[i][0]>=t):
	# 		r = i
	# 		break
	#binary search
	high = len(arr)
	low = 0
	while low + 1 <high:
		mid = (high + low)//2
		if(arr[mid][0]>=t):
			high = mid
		else:
			low = mid
	# if (high!=r):
	# 	print("binary search wrong", high, r)
	# 	sys.exit()
	r= high
	left = arr[r-1]
	right = arr[r]

	dlat = right[1] - left[1]
	dlon = right[2] - left[2]
	

	dx = right[0] - left[0]
	x = t-left[0]
	
	lat = left[1] + x/dx * dlat
	lon = left[2] + x/dx * dlon

	return  lat, lon

def totime(timestring):
    return datetime.strptime(timestring, timestamp_format)


def break_trajectory(vessels):

    for mmsi, vesselrows in vessels.items():
        breakpoints = [0]
        for i in range(1, len(vesselrows)):
            # 1.if vesselrows[i-1] is far back in time from vesselrows[i]
            # 2.and extrapolation from vesselrows[i-2] [i-1] to [i] takes it out of map
            # 3.and extrapolation from vesselrows[i+1] [i] to [i-1] takes it out of map
            # 4.and vesselrows[i-1] and vesselrows[i] are near boundary
            #   then it's a good idea to break trajectory.

            confidence = 0
            #1
            if(vesselrows[i][0] - vesselrows[i-1][0] > timedelta(hours=2)):
                confidence += 1
            
            
            #2
            if(i-2 >= 0):
                diff = vesselrows[i-1][2] - vesselrows[i-2][2]
                extrapolated = vesselrows[i-1][2] + diff
                if(extrapolated > maxlon or extrapolated < minlon): # going east or west respectively
                    confidence +=1
                    # print("departure ",extrapolated, vesselrows[i-1]+2)
            
            #3
            if(i+1 < len(vesselrows)):
                diff = vesselrows[i+1][2] - vesselrows[i][2]
                extrapolated = vesselrows[i][2] - diff
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

###-----------------------------------------------------------------------

#------------------------------------------------------ print unique columns
count_dict = {}
for i in range(datalen):
	value_of_interest = data[attribute][i]
	if(value_of_interest in count_dict):
		count_dict[value_of_interest] += 1
	else:
		count_dict[value_of_interest] = 1

sorted_arr = sorted([(val, key) for key, val in count_dict.items()])

for i in sorted_arr:
	print(i)
#------------------------------------------------------Heading-course analysis
heading_analysis = False
if(heading_analysis):
	speeds = []
	hcd = [] # hcd = heading-course difference
	countt=0
	for i in range(1,datalen):
		# check if timestamp is sorted
		timestamp = datetime.strptime(data["TIMESTAMP"][i], timestamp_format)
		prevtimestamp = datetime.strptime(data["TIMESTAMP"][i-1], timestamp_format)
		if (timestamp < prevtimestamp):
			print("ERROR: timestamp at row ", i-1, " = ", prevtimestamp," is larger than ")
			print(" timestamp at row", i, " = ", timestamp)
			break
		
		# speed - (course-heading) relation check
		if(511 != data['HEADING'][i] and None!=data['HEADING'][i]):
			xvalue = float(data['SPEED (KNOTSx10)'][i])
			speeds.append(xvalue)
			diff = float(data['COURSE'][i]) - float(data['HEADING'][i])
			diff = (abs(diff) + 360)%360
			if(diff>180):
				diff = 360-diff	
			hcd.append(diff)

			if(xvalue>100 and diff>70 and diff <110):
				countt += 1
				print(i, xvalue, diff)
				print(data['MMSI'][i], data['TIMESTAMP'][i], data['HEADING'][i], data['COURSE'][i])
				print("------")
				if(countt>50):
					break

	# special analysis
	
	for i in range(len(speeds)):
		if(speeds[i]>120 and hcd[i]>70 and hcd[i]<110):
			countt += 1
			print(i, speeds[i], hcd[i])
			if(countt>50):
				break
	print("anamolous speed vs (course-heading) data point count: " ,countt)

	plt.scatter(speeds,hcd, c='blue', s=4, alpha=0.5)
	plt.xlabel('speed in knots/10')
	plt.ylabel('course-heading (in degrees)')
	plt.yticks([90*x for x in range(0,3)])
	plt.show()
	sys.exit()
#------------------------------------------------------Vessel time difference analysis

# Vessel timestamp difference between subsequent points analysis
# get {mmsi:[timestamp1,timestamp2]} for vessel
vesselAnalysis = False
if(vesselAnalysis):
	vesseltimes = {}
	vesselvalues={}
	

	
	for i in range(datalen):
		mmsi = data["MMSI"][i]
		time_obj = datetime.strptime(data["TIMESTAMP"][i], timestamp_format)
		# rowvalue = (time_obj, data["LAT"][i] , data["LON"][i], data["HEADING"][i], getVesselColor())
		value_of_interest = data[attribute][i] # change this for your analysis
		if(mmsi in vesseltimes):
			vesseltimes[mmsi].append(time_obj)
			vesselvalues[mmsi].append(value_of_interest)
		else:
			vesseltimes[mmsi]=[time_obj]
			vesselvalues[mmsi]=[value_of_interest]

	diffs = []
	values = []
	for mmsi in vesseltimes.keys():
		for i in range(1, len(vesseltimes[mmsi])):
			diffs.append((vesseltimes[mmsi][i] - vesseltimes[mmsi][i-1]).total_seconds()/60.0)
			values.append(vesselvalues[mmsi][i]) # TODO: or should it be i-1

	# plt.scatter(diffs, values, c='red', s=4, alpha=0.5)
	# plt.ylabel(attribute)
	# plt.xlabel('time difference between two datapoints of ship in minutes')
	# plt.show()

	uniq_vals = list(set(values))
	iddx = {}
	for i,uval in enumerate(uniq_vals):
		iddx[uval] = i

	bucket_size = 1
	outliercount = 0
	bbb = np.zeros(shape=(len(uniq_vals), bucket_count))
	diffbuckets = [x for x in range(bucket_count)]
	for diffidx, diff in enumerate(diffs):
		idx = int(diff/bucket_size)
		if(idx<bucket_count):
			bbb[iddx[values[diffidx]]][idx] +=1
		else:
			outliercount+=1
	# print(bbb)
	print("outlier count:", outliercount)
	for ngraph,val in enumerate(bbb):
		plt.bar(diffbuckets,val,label=uniq_vals[ngraph])
		if(ngraph>30): #TODO: remove this
			break
	plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0., title=attribute)
	plt.xlabel('time difference between two datapoints of ship in minutes')
	plt.ylabel('number of data points')

	plt.show()
#------------------------------------------------------Traffic intensity by zone
# (i.e number of ships in a zone against timestamp)

# get pointers to all 25 zones
cols = ['r','g','b','c','m','y','k']
styles = ['-','--','-.',':','','','']
fig, ax = plt.subplots(figsize=(16,9))


# Go through the timesteps while checking which zones a lat lon point lies in

exact_starttime = datetime.strptime(data["TIMESTAMP"][0], timestamp_format) 
exact_endtime = datetime.strptime(data["TIMESTAMP"][datalen-1], timestamp_format) 

starttime = nextnear(exact_starttime, interval) 
endtime = prevnear(exact_endtime, interval) 

# get {mmsi:[timestamp1,timestamp2]} for vessel
vessels = {}
for i in range(datalen):
	if(i%10000 ==0):
			print("data row ",i,"/",datalen)
	mmsi = data["MMSI"][i]
	time_obj = datetime.strptime(data["TIMESTAMP"][i], timestamp_format)
	rowvalue = (time_obj, data["LAT"][i] , data["LON"][i])
	if(mmsi in vessels):
		vessels[mmsi].append(rowvalue)
	else:
		vessels[mmsi]=[rowvalue]
# -------------------Use the above data in multiple places

# get trajectory from vessels.
break_trajectory(vessels)


zoneFile = opt.zone_file
gdf = gpd.GeoDataFrame.from_file(zoneFile)
polygons = gdf.geometry.values
poly_count = len(polygons)


entry_exit_check = False
if(entry_exit_check):
	for mmsi, vesseltimes in vessels.items():
		for i in range(1, len(vesseltimes)):
			if(vesseltimes[i][0] - vesseltimes[i-1][0] > timedelta(hours=opt.exit_entry_diff)):
				print(mmsi, ": FROM ",vesseltimes[i-1], "\tTO", vesseltimes[i], "\t: DIFF : ", vesseltimes[i][0] - vesseltimes[i-1][0], end='\t')
				print()

# populate the [[{mmsi:(lat,lon,heading)}, {mmsi:(lat,lon,heading)}]]
# populate the [clock_time										  ]	 for above
if(opt.zone_traffic):
	interpolatedLatlongs = []
	timesteps = []

	timestep = starttime
	while timestep != endtime:
		vessel_timestep_map = {}
		for mmsi, vesseltimes in trajectories.items():
			if (timestep > vesseltimes[0][0] and timestep < vesseltimes[-1][0]):
				lat, lon =interpolate(vesseltimes, timestep)
				vessel_timestep_map[mmsi] = (lat, lon) # TODO: add heading
		interpolatedLatlongs.append(vessel_timestep_map)
		timesteps.append(timestep)

		timestep = timestep + timedelta(minutes=interval)
	# use timesteps and corresponding interpolatedlatlongs from here

	show_tot_ships = False
	if(show_tot_ships):
		total_ships_per_timestamp = [len(ships_at_t) for ships_at_t in interpolatedLatlongs]
		plt.plot(timesteps, total_ships_per_timestamp, label='TOTAL ships')
		plt.show()

	zones_to_plot = [x for x in range(64)]
	# zones_to_plot = [22,25]
	int_plot = np.zeros(shape=(len(polygons), len(timesteps)))
	for t, val in enumerate(interpolatedLatlongs):
		if(t%10000 ==0):
			print("timestep ",t,"/",len(interpolatedLatlongs))
		ship_cur_positions = []
		for mmsi, vessel in val.items():
			# print("vessel: ", vessel)
			latitude = vessel[0]
			longitude = vessel[1]
			ship_cur_positions.append(Point(longitude, latitude))
		shp_pnts = gpd.GeoDataFrame(geometry=ship_cur_positions)
		# print(shp_pnts)
		for pind in zones_to_plot:
			# print(polygon)
			vals = shp_pnts.within(polygons[pind])
			for pnt_key, is_inside in enumerate(vals):
				if(is_inside):
					# print("FOUND INSIDE ",pind, t)
					plotline_ind = pind if opt.separate_zone_plot else 0
					int_plot[plotline_ind][t] +=1
		# sys.exit()
	# plot the traffic against time

	#incase of daily_bins
	bins_in_day = 24
	bin_duration = 60 * bins_in_day//24
	nums_in_bin = bin_duration//interval
	print("nums_in_bin", nums_in_bin)
	xticks = [x for x in range(bins_in_day)]

	lines_to_plot = zones_to_plot if opt.separate_zone_plot else [0]
	for pind in lines_to_plot:
		for i in range( len(int_plot[pind]) , opt.lastnavg, -1):
			int_plot[pind][i-1]=sum(int_plot[pind][i-opt.lastnavg:i])/opt.lastnavg
		
		if(opt.daily_bins):
			# plot both mean and variance.
			binmeans = [0] * bins_in_day
			binsds = [0] * bins_in_day
			for i in range(bins_in_day):
				summ = 0
				sumsqrdiff = 0
				count = 0
				for j in range(i*nums_in_bin, len(int_plot[pind]), bins_in_day * nums_in_bin):
					count += 1
					summ += max([int_plot[pind][k] for k in range(j,min(j+nums_in_bin,len(int_plot[pind])))])
					# summ +=int_plot[pind][j+k] # should be a max out of 60/interval points.(where interval = 1 ideally)
				mean = summ/count if count > 0 else 0
				for j in range(i*nums_in_bin, len(int_plot[pind]), bins_in_day * nums_in_bin):
					sumsqrdiff += (max([int_plot[pind][k] for k in range(j,min(j+nums_in_bin,len(int_plot[pind])))]) - mean) ** 2
				binmeans[i] = mean
				binsds[i] = math.sqrt(sumsqrdiff)
			plt.plot(xticks, binmeans,cols[(pind//4)%7]+styles[pind%4], label='mean ships')
			plt.plot(xticks, [m+sd for m,sd in zip(binmeans, binsds)] ,'g--', label='mean + SD')
			plt.plot(xticks, [m-sd for m,sd in zip(binmeans, binsds)] ,'g--', label='mean - SD')
		else:
			plt.plot(timesteps, int_plot[pind],cols[(pind//4)%7]+styles[pind%4], label=str(pind))
		# if(i>=28):
		# 	break
	plt.legend(bbox_to_anchor=(1.0, 1.1), loc='upper left', title="zone_id", borderaxespad=0.)
	plt.xlabel('hour of the day ('+str(opt.interval) + ' minute interval)')
	plt.ylabel('number of ships in tss zones')
	plt.show()

#-------------------------- plot polygon numbers
if(opt.show_map):
	ax.set_xlim(103.535, 104.05)
	ax.set_ylim(1.02, 1.32)


if(opt.show_map):
	gdf.plot(ax=ax, color=(cols*((poly_count+len(cols))//len(cols)))[:poly_count])
	
	for pind, polygon in enumerate(polygons):
		
		# sys.exit()
		pind_text = plt.text(polygon.centroid.x, polygon.centroid.y, str(pind),color='w')
	
	# mmsi_of_interest = 563034560
	mmsi_of_interest = 312471000
	lats = [data_point[1] for data_point in vessels[mmsi_of_interest]]
	lons = [data_point[2] for data_point in vessels[mmsi_of_interest]]
	# print(lats)
	# plt.plot([1.023,1.024],[103.6, 103.7], label = 'ship route', color='r')
	lll = 200
	plt.plot(lons[:lll], lats[:lll], 'b')
	plt.plot(lons[lll-1:], lats[lll-1:], 'r')

	plt.show()