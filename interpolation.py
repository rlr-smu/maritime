from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

input_file = 'one_Vessel_Position_raw.csv'
timestamp_format = '%Y-%m-%d %H:%M:%S'


headers = []
rows = []
for lineNo,line in enumerate(open(input_file, 'r').readlines()):
    fields = [field.strip() for field in line.split(',')]
    if(lineNo==0):# collect headers
        headers = fields
        continue
    if(lineNo>100):
        break
    
    row = {}
    for header, value in zip(headers, fields):
        row[header] = value
    rows.append(row)

lons = [float(row['LON'])-103.0 for row in rows] # TODO: remove 103
lats = [float(row['LAT']) for row in rows]
clocktimes = [datetime.strptime(row['TIMESTAMP'], timestamp_format) for row in rows]

# print(lons)
ts = [ t.timestamp() for t in clocktimes]
# print(xs)
# plt.plot(ts,lons)
# plt.plot(ts,lats,'r-')
plt.plot(ts,lats,'r.')

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

print(newts)
print(newlats)
plt.plot(newts, newlats,'g.')

red_patch = mpatches.Patch(color='red', label='raw data')
green_patch = mpatches.Patch(color='green', label='interpolated data')
plt.xlabel('seconds since 1970')
plt.ylabel('latitude of ship')
plt.legend(handles=[red_patch, green_patch])
plt.show()