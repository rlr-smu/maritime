from datetime import datetime, timedelta
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--latlonfile",  default='data/two_ships.csv', help='file with ship location data(AIS)') 
parser.add_argument("--rows_to_read", type=int,  default=1000, help="limit the number of rows from latlonfile") 
parser.add_argument("--column_index", type=int,  default=6, help="0. MMSI,VESSEL TYPE,LENGTH,WIDTH,IMO,5.DESTINATION,STATUS,SPEED (KNOTSx10),LON,LAT,10.COURSE,HEADING,TIMESTAMP") 
parser.add_argument("--bucket_count", type=int,  default=100, help="total buckets.") 

opt = parser.parse_args()
print(opt)

vessel_csv_filename=opt.latlonfile
rows_to_read = opt.rows_to_read
bucket_count = opt.bucket_count
timestamp_format = '%Y-%m-%d %H:%M:%S'

data = pd.read_csv(vessel_csv_filename)
datalen = min(rows_to_read, len(data["MMSI"]))

#------------------------------------------------------Heading-course analysis
heading_analysis = False
if(heading_analysis):
	speeds = []
	hcd = [] # hcd = heading-course difference
	for i in range(1,datalen):
		# check if timestamp is sorted
		timestamp = datetime.strptime(data["TIMESTAMP"][i], timestamp_format)
		prevtimestamp = datetime.strptime(data["TIMESTAMP"][i-1], timestamp_format)
		if (timestamp < prevtimestamp):
			print("ERROR: timestamp at row ", i-1, " = ", prevtimestamp," is larger than ")
			print(" timestamp at row", i, " = ", timestamp)
			break
		
		# speed - (course-heading) relation check
		if(511 != data['HEADING'][i]):
			speeds.append(float(data['SPEED (KNOTSx10)'][i]))
			diff = float(data['COURSE'][i]) - float(data['HEADING'][i])
			diff = (abs(diff) + 360)%360	
			hcd.append(diff)
	plt.scatter(speeds,hcd, c='blue', s=4, alpha=0.5)
	plt.xlabel('speed in knots/10')
	plt.ylabel('course-heading (in degrees)')
	plt.yticks([90*x for x in range(-4,5)])
	plt.show()

#------------------------------------------------------Vessel timestamp analysis

# Vessel timestamp difference between subsequent points analysis
# get {mmsi:[timestamp1,timestamp2]} for vessel
vesseltimes = {}
vesselvalues={}
columns = ['MMSI', 'VESSEL TYPE', 'LENGTH', 'WIDTH', 'IMO', 'DESTINATION', 'STATUS', 'SPEED (KNOTSx10)', 'LON', 'LAT', 'COURSE', 'HEADING', 'TIMESTAMP']

attribute = columns[opt.column_index]
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