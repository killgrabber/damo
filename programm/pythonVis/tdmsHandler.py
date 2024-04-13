from nptdms import TdmsFile
import os

import numpy as np
import matplotlib.pyplot as plt

CHANNEL_NAMES = ["Ch2_Sen2_neu", "Ch1_Sen1_neu",  "Ch3_Tast_neu",  

                ]

stringLine = \
    "name: " + \
    ", date " + \
    ", time " + \
    ", length " + \
    ", powerCombined(N) " + \
    ", power1(N) " + \
    ", power2(N) " + \
    ", index "

pois = []
pois.append(stringLine)

def getIndexOfSettledAfterMax(array, MAX_GRADIENT=0.0009):
    index_max = np.argmax(array)
    gradientPowerCombined = getGradients(array)
    #find index when graph has settled:
    while index_max < len(array):
        #print(abs(gradientPowerCombined[index_max]))
        if(abs(gradientPowerCombined[index_max]) < MAX_GRADIENT):
            return index_max
        index_max +=1
    return -1
            
def showPlot(channels):
    fig, ax1 = plt.subplots()
    #channel 1, 2: NM channel 3; mm
    color = 'tab:red'
    ax1.set_xlabel('time (ms)')
    ax1.set_ylabel('N', color=color)
    #ax1.plot(channels[0], color=color, label="Kraft Achse 1")
    color = 'tab:orange'
    #ax1.plot(channels[1], color=color, label="Kraft Achse 2")
    ax1.tick_params(axis='y', labelcolor=color)    

    combined = []  #combined force
    for index in range(len(channels[0][:])):
        combined.append(channels[0][index] + channels[1][index])
    ax1.plot(combined, color="tab:pink", label="Kraft Kombiniert")
    ax1.legend(loc=2)    

    index = getIndexOfSettledAfterMax(combined)
    index_max = np.argmax(combined)
    ax1.annotate('max',
            xy=(index_max, combined[index_max]), xycoords='data',
            xytext=(-10, 90), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='center', verticalalignment='bottom')

    ax1.annotate('max + settled',
            xy=(index, combined[index]), xycoords='data',
            xytext=(10, -90), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='center', verticalalignment='bottom')
    #ax1.plot(gradientPowerCombined, color="tab:brown", label="Gradient Kraft")



    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('mm', color=color)
    ax2.plot(channels[2], color=color, label="Taster Verschiebung")
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc=1)

    time = str(channels[0].properties['wf_start_time'])
    date = time.split("T")[0]
    timeStr = time.split("T")[1].split(".")[0]
    stringLine = \
    str(file.split(".")[0]) + \
    ", "+ str(date) + \
    ", "+ str(timeStr) + \
    ", "+ str(channels[2][index]) + \
    ", " + str(combined[index]) + \
    ", "+ str(channels[0][index]) + \
    ", " + str(channels[1][index]) + \
    ", "+ str(index)
    
    pois.append(stringLine)
    plt.title(file)
    fig.tight_layout()
    #plt.show()

def getGradients(channel):
    return np.gradient(channel[:])


for file in os.listdir("daten/tdms/"):
    if file.endswith(".tdms"):
        #print(os.path.join("daten/tdms", file))
        tdms_file = TdmsFile.read(os.path.join("daten/tdms", file))
        group = tdms_file['Log']
        allChannels = []
        for channelName in CHANNEL_NAMES:
            try:
                allChannels.append(group[channelName])
            except KeyError:
                print("Channel: " + channelName + " not found")
        showPlot(allChannels)

        f = open("tdms2exec.txt", "w")
        for poi in pois:
            f.write(poi + "\n")
        f.close()

