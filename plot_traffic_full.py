import pickle
import pandas as pd
from collections import defaultdict
from matplotlib import pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--fname', default='what.pkl')

args = parser.parse_args()


traffic_datas={}
zoneids = [i for i in range(13)]+ [i for i in range(14,20)]
# zoneids = [i for i in range(6)]
for zoneid in zoneids:
    print('loading zone', zoneid, '...')
    filename ='aamas_2yrdata/bulktraffic_data_' + str(zoneid) + '_838357.pkl'
    with open(filename, 'rb') as f:
        traffic_datas[zoneid] = pickle.load(f)

print('summing traffic data up...')
sum_traffic_data = {}
for ts in traffic_datas[5]: # zone 5 is expected to have all timestamps
    sum_traffic_data[ts] = sum([traffic_datas[zoneid][ts] for zoneid in zoneids])


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

    np_avgs = []
    np_stds = []
    meanplussds = []
    meanminussds = []
    for traffic_minute,traffic_counts in sorted(daily_traffic.items()):
        lenn = len(traffic_counts)
        if(lenn > 0 ):
            xs.append(int(traffic_minute))
            lens.append(lenn)
            traffic_counts = sorted(traffic_counts)
            tc_np = np.array(traffic_counts)
            np_avgs.append(np.mean(tc_np))
            np_stds.append(np.std(tc_np))
            meanplussds.append(np.mean(tc_np)+np.std(tc_np))
            meanminussds.append(np.mean(tc_np)-np.std(tc_np))
            medians.append(traffic_counts[lenn//2])
            perc_25.append(traffic_counts[lenn*32//100])
            perc_75.append(traffic_counts[lenn*68//100])
            avgs.append(float(sum(traffic_counts)/ len(traffic_counts)))
            # print("len(traffic_counts)", len(traffic_counts))
    print(xs)
    # print(avgs)
    with open('real_avg_daily_traffic.pkl', 'wb') as f:
        pickle.dump(avgs, f)
    # plt.plot([8],[0],'b.')
    # plt.axis('scaled')
    currentzoneid = args.fname
    plt.plot(xs, avgs, label='avg traffic')
    # plt.plot(xs, lens, label='lens '+str(currentzoneid))
    plt.plot(xs, medians, label='medians ')
    plt.plot(xs, perc_25, label='perc32 ')
    plt.plot(xs, perc_75, label='perc68 ')
    plt.plot(xs, np_avgs, label='np_avg ')
    plt.plot(xs, np_stds, label='np_std ')
    plt.plot(xs, meanplussds, label='meanplussds ')
    plt.plot(xs, meanminussds, label='meanminussds ')

    plt.xlabel('minute of the day(upto 1440)')
    plt.ylabel('traffic in the zone '+str(zoneids))
    plt.legend()
    plt.show()
print('plotting..')
plot_daily_traffic(sum_traffic_data)