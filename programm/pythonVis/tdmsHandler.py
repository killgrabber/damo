from nptdms import TdmsFile
import os

import numpy as np
import matplotlib.pyplot as plt

CHANNEL_NAMES = ["Ch2_Sen2_neu", "Ch1_Sen1_neu", "Ch3_Tast_neu"]

stringLine = \
    "name: " + \
    ", date " + \
    ", time " + \
    ", length " + \
    ", powerCombined(N) " + \
    ", power1(N) " + \
    ", power2(N) " + \
    ", index "

pois = [stringLine]


def getIndexOfSettledAfterMax(array, max_gradient=0.0009):
    index_max = np.argmax(array)
    gradientPowerCombined = getGradients(array)
    # find index when graph has settled:
    while index_max < len(array):
        # print(abs(gradientPowerCombined[index_max]))
        if abs(gradientPowerCombined[index_max]) < max_gradient:
            return index_max
        index_max += 1
    return -1


def showPlot(channels, name, save_dir):
    plt.rc('font', size=20)
    plt.grid()
    fig, ax1 = plt.subplots()
    # channel 1, 2: NM channel 3; mm
    color = 'tab:red'
    ax1.set_xlabel('Zeit (ms)')
    ax1.set_ylabel('Spannkraft [N]', color=color)
    # ax1.plot(channels[0], color=color, label="Kraft Achse 1")
    color = 'tab:orange'
    # ax1.plot(channels[1], color=color, label="Kraft Achse 2")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid()
    combined = []  # combined force
    for index in range(len(channels[0][:])):
        combined.append(channels[0][index] + channels[1][index])
    ax1.plot(combined, color="tab:pink", label="Kraft Kombiniert")
    #ax1.legend(loc=2)
    ax1.legend(bbox_to_anchor=(0.0, 1.2), loc="upper left")

    index = getIndexOfSettledAfterMax(combined)
    nm_value_settled = combined[index]
    taster_value_sattled = channels[2][index]
    index_max = np.argmax(combined)
    #ax1.annotate('max',
    #             xy=(index_max, combined[index_max]), xycoords='data',
    #             xytext=(-10, 90), textcoords='offset points',
    #             arrowprops=dict(facecolor='black', shrink=0.05),
    #             horizontalalignment='center', verticalalignment='bottom')

    ax1.annotate(f'settled\n{nm_value_settled:.3f}nm\n{taster_value_sattled:.3f}mm',
                 xy=(index, combined[index]), xycoords='data',
                 xytext=(10, -90), textcoords='offset points',
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 horizontalalignment='center', verticalalignment='bottom')
    # ax1.plot(gradientPowerCombined, color="tab:brown", label="Gradient Kraft")

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Verschiebung [mm]', color=color)
    ax2.plot(channels[2], color=color, label="Taster")
    ax2.grid()
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(bbox_to_anchor=(0.5, 1.2), loc="upper left")
    time = str(channels[0].properties['wf_start_time'])
    date = time.split("T")[0]
    timeStr = time.split("T")[1].split(".")[0]
    stringLine = \
        str(name) + \
        ", " + str(date) + \
        ", " + str(timeStr) + \
        ", " + str(channels[2][index]) + \
        ", " + str(combined[index]) + \
        ", " + str(channels[0][index]) + \
        ", " + str(channels[1][index]) + \
        ", " + str(index)

    pois.append(stringLine)
    plt.title(name)
    fig.tight_layout()
    plot_dir = save_dir + "/plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_name = plot_dir + "/" + name + ".png"
    #plt.show()
    plt.savefig(plot_name)


    return nm_value_settled, taster_value_sattled


def getGradients(channel):
    return np.gradient(channel[:])


def get_combined_plot(textfiles):
    all_nm_values = []
    all_taster_values = []
    for textfile in textfiles:
        file = open(textfile)
        nm_value_settled = []
        taster_value_settled = []
        for line in file:
            splitted = line.split(',').copy()
            nm = float(splitted[1])
            if nm >= 0:
                mm = float(splitted[2])
                if len(nm_value_settled) > 0:
                    nm += nm_value_settled[-1]
                if len(taster_value_settled) > 0:
                    mm += taster_value_settled[-1]
                nm_value_settled.append(nm)
                taster_value_settled.append(mm)
        file.close()
        all_nm_values.append(nm_value_settled)
        all_taster_values.append(taster_value_settled)

    plt.rc('font', size=40)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Spannkraft und Verschiebung')
    colors = ['b', 'g', 'r', 'c', 'm', 'y', ]
    for index in range(len(all_taster_values)):
        filename = textfiles[index].split('/')[-1]
        labelname = filename.split('_')[0] + filename.split('_')[1]
        ax1.plot(all_nm_values[index], '--x', label=f"{labelname}",
                 color=colors[(index + 3) % len(textfiles)])
        ax2.plot(all_taster_values[index], '--o', label=f"{labelname}",
                 color=colors[(index + 3) % len(textfiles)])

    ax1.legend()
    ax2.legend()
    ax1.set_ylabel('Spannkraft [nm]')
    ax2.set_ylabel('Verschiebung [mm]')
    ax1.set_xlabel('Spannkraftstufe')
    ax2.set_xlabel('Spannkraftstufe')
    plt.legend()
    ax1.grid()
    ax2.grid()
    plt.show()


def getTDMSdata(dir: str):
    f = open(dir + "_all.txt", "w")
    f.write(f"first line, {0}, {0}\n")
    print(f"Dir name: {dir}")
    for file in os.listdir(dir):
        if file.endswith(".tdms"):
            # print(os.path.join("daten/tdms", file))
            print(f"Trying to open tdms: {os.path.join(file)}")
            tdms_file = TdmsFile.read(dir + "/" + file)
            group = tdms_file['Log']
            name = tdms_file.properties["name"]
            allChannels = []
            for channelName in CHANNEL_NAMES:
                try:
                    allChannels.append(group[channelName])
                except KeyError:
                    print("Channel: " + channelName + " not found")
            nm_value_settled, taster_value_sattled = showPlot(allChannels, name, dir)
            f.write(f"{file}, {nm_value_settled}, {taster_value_sattled}\n")
    f.close()
    print("Done converting tdms plots")
