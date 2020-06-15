from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import math

input_file = 'data/one_Vessel_Position_raw.csv'
# input_file = 'data/zero_imo.csv'
# input_file = 'data/mmsi0.csv'
# input_file = 'data/singleshipB.csv'
# input_file = 'data/singleshipA.csv'
# input_file = 'data/two_ships.csv'
# input_file = 'data/first_ship.csv'
timestamp_format = '%Y-%m-%d %H:%M:%S'

ref_point = (1.258, 103.771) # singapore port coordinates as the reference point
km_per_lat = 111.321 # kilometers per latitude

headers = []
rows = []
for lineNo,line in enumerate(open(input_file, 'r').readlines()):
    fields = [field.strip() for field in line.split(',')]
    if(lineNo==0):# collect headers
        headers = fields
        continue
    if(lineNo>1000):
        break
    
    row = {}
    for header, value in zip(headers, fields):
        row[header] = value
    rows.append(row)

lons = [float(row['LON'])-ref_point[1] for row in rows] # TODO: remove 103
lats = [float(row['LAT'])-ref_point[0] for row in rows]
headings = [float(row['HEADING'])/10 for row in rows]
courses = [float(row['COURSE'])/10 for row in rows]
speeds = [float(row['SPEED (KNOTSx10)'])/10 for row in rows]
statuses = [int(row['STATUS']) for row in rows]

# convert lat lons to km TODO: this might screw up interpolation.. careful!
to_km = True
yaxis_label = 'difference in lat/long of the ships from singapore port'
if(to_km):
    lats = [x * km_per_lat for x in lats]
    lons = [x * km_per_lat for x in lons]
    yaxis_label = 'distance from singapore port in km'
clocktimes = [datetime.strptime(row['TIMESTAMP'], timestamp_format) for row in rows]

# print(lons)
ts = [ t.timestamp() for t in clocktimes]
# print(xs)
plt.plot(clocktimes,lons,'b.')
plt.plot(clocktimes,lats,'r.')
plt.plot(clocktimes, headings, 'c.')
plt.plot(clocktimes, courses, 'y.')
plt.plot(clocktimes, speeds, 'm.')
plt.plot(clocktimes, statuses, 'ko')

cnt = 150 # output evenly spaced point count ( first point will be lats[0] and last one will be lats[-1])
newts = np.linspace(ts[0], ts[-1], cnt)

#newlats = [(lats[-1]-lats[0])*(newt - ts[0])/(ts[-1]-ts[0])+lats[0] for newt in newts]

# brute force newlats
newlats = [lats[0]]
for newt in newts[1:-1]:
    for i in range(len(ts)):
        if(ts[i] >= newt):
            prev = i-1
            nex = i
            dt = ts[nex] - ts[prev]
            dlat = lats[nex] - lats[prev]
            newlat = lats[prev] + (newt - ts[prev]) * dlat/dt
            newlats.append(newlat)
            break
newlats.append(lats[-1])

# print(newts)
# print(newlats)
# plt.plot(newts, newlats,'g.')

red_patch = mpatches.Patch(color='red', label='latitude')
blue_patch = mpatches.Patch(color='blue', label='longitude')
green_patch = mpatches.Patch(color='green', label='interpolated data')
yellow_patch =  mpatches.Patch(color='yellow', label='course in degrees x10')
cyan_patch =  mpatches.Patch(color='cyan', label='heading in degrees x10')
magenta_patch =  mpatches.Patch(color='magenta', label='speed in knots ')
black_patch =  mpatches.Patch(color='black', label='Status(integer)')

plt.xlabel('time')
plt.ylabel(yaxis_label)
plt.legend(handles=[red_patch, blue_patch, yellow_patch, cyan_patch, magenta_patch, black_patch])
plt.show()