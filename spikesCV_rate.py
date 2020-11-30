import neo
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, lfilter
from scipy.signal import find_peaks
from scipy.signal import peak_widths
import numpy as np
import csv
import scipy.stats as st
from matplotlib.offsetbox import AnchoredText

# parameters to pay attention in the script highlighted with *****
# filepath, file_start, file_end,
# filter_pandas_series - filter order,
# Histogram with bins
# also without_first_N_samples = df.iloc[360:] deleting artefacts in the beginning of data
#****************************************************************************
filepath = '/home/murat/Documents/Data_Recorded_CNRS/2020_05_27/20200527_Cell407_001.smr'
artifact_threshold = 0.2
spike_threshold = 3
#    range of analysing data
file_start = 235  # seconds
file_end = 237  # seconds
hide_stim_artifacts = True
spikelets_thresh = 0.5
#******************************************************************************

def get_spike_2(file):
    reader = neo.io.Spike2IO(filename=file)
    # # read the block
    bl = reader.read(lazy=False, load_waveforms=True)[0]
    for seg in bl.segments:
        print("SEG: " + str(seg.file_origin))
        for asig in seg.analogsignals:
            if 'Vm' in asig.name:
                vm_samples = asig.shape[0]
                vm_series = pd.Series(
                    data=asig.magnitude.reshape(vm_samples),
                    index=asig.times.rescale('s').magnitude.reshape(vm_samples),
                    name='Vm'
                )
            if 'Channel bundle' in asig.name:
                # bl.segments[0].analogsignals[2]
                analog_samples = asig.shape[0]
                analog_channels = asig.shape[1]
                analog_df = pd.DataFrame(index=asig.times.rescale('s').magnitude.reshape(analog_samples))
                channel_names = asig.name[(asig.name.find('('))+1:(asig.name.find(')'))].split(',')
                for i in range(analog_channels):
                    analog_df[channel_names[i]] = asig[:, i].magnitude

                # data=[asig[:,0].magnitude, asig[:,1].magnitude], )
    print(vm_series.index.max(), "s, in this recording")
    return vm_series.loc[file_start:file_end], analog_df.loc[file_start:file_end]


def filter_pandas_series(data_series, cutoff, type='high', order=5): # use array for type='band' [low, high]
    min_fs = np.round(data_series.reset_index()['index'].diff().min(), 8)
    max_fs = np.round(data_series.reset_index()['index'].diff().max(), 8)
    if min_fs == max_fs:
        fs = 1 / min_fs
    else:
        print('cannot filter non-continuous data: try to upsample first')
    nyq = 0.5 * fs
    cutoff /= nyq
    b, a = butter(order, cutoff, btype=type, analog=False)
    y = lfilter(b, a, data_series.values)
    return pd.Series(y, data_series.index)

vm, channels = get_spike_2(filepath)
vm_filt = filter_pandas_series(vm, 100, type='high')

#       converting file to .csv format
#    ----------------------------------------------------------------------
#plt.plot(vm_filt)
df = pd.DataFrame(vm_filt)
without_first_N_samples = df.iloc[220:]  # deleting first N samples in dataframe because of artefact in the beginning
#plt.plot(without_first_N_samples)

# saving the dataframe as .csv
without_first_N_samples.to_csv('/home/murat/Desktop/file1.csv')
with open('/home/murat/Desktop/file1.csv') as file:
    reader = csv.reader(file, delimiter=',')
sig = pd.read_csv('/home/murat/Desktop/file1.csv')
#    ---------------------------------------------------------------------

#  creating data table and finding peaks from it

signals = pd.DataFrame(sig)
signals.columns = ['time', 'Vm']
#print(signals)

peaks, _ = find_peaks(signals['Vm'], height=(0, 7), prominence=7)
peaks_table = pd.DataFrame(peaks)
width_half = peak_widths(signals['Vm'], peaks, rel_height=0.25)

#       for wdh dataframe [0]-row is half width; [1]-row is level of the half width; [2]-?; [3]-?
wdh = pd.DataFrame(width_half)
width_half[0]
width_full = peak_widths(signals['Vm'], peaks, rel_height=1)
width_full[0]
# plt.plot(signals['Vm'])
# plt.plot(peaks, signals['Vm'][peaks], "*")
# plt.hlines(*width_half[1:], color="C2")
# plt.hlines(*width_full[1:], color="C3")
# plt.show()


# creating table of spikes with corresponding time, amplitude, samples
rows = []
for it in peaks:
    rows.append([it, signals._get_value(index=it, col='Vm'), signals._get_value(index=it, col='time')])

dh = pd.DataFrame(rows, columns=["samples", "Vm", "time"])
number_of_ISI = dh.index.max() - 1
firing_rate = number_of_ISI / (file_end - file_start)
print('firing_rate =', firing_rate, 'Hz')



#  ------------------------------------------------------------------------------
# calculating interspikes intervals (ISI) and coefficient of variance (CV),
intervals = []
for i in range(0, dh.index.max()):
    intervals.append([i, dh._get_value(index=i+1, col='time') - dh._get_value(index=i, col='time')])

ISI_table = pd.DataFrame(intervals, columns=["N", "ISI"])
spike_intervals = np.array(intervals)[:, 1]
mean_of_ISI = np.mean(spike_intervals)
stdev_intervals = np.std(spike_intervals, ddof=1)   # standard deviation (SD) of intervals;  ddof=1 for case 1/(N-1),  ddof=0 for case 1/N
coef_var = stdev_intervals/mean_of_ISI
sterr = stdev_intervals / np.sqrt(spike_intervals.size)   #  standard error of the mean (SEM)
print('coefficient of variance =', coef_var)

#  --------------------------------------------------------------------------------

# calculating spike widths at half peak; creating table of spikes_widths, with corresponding samples
wdh_0 = np.array(width_half[0])
widathalf = pd.DataFrame(wdh_0, columns=["width"])
widths_rows = []
for i in range(0, widathalf.index.max()):
    widths_rows.append([i, widathalf._get_value(index=i, col='width')])

spikes_widths = np.array(widths_rows)[:, 1]
number_of_all_widths = widathalf.index.max()
mean_of_spikes_widths = np.mean(spikes_widths)
stdev_widths = np.std(spikes_widths, ddof=1)  # standard deviation (SD) of intervals;  ddof=1 for case 1/(N-1),  ddof=0 for case 1/N
coef_var_width = stdev_widths/mean_of_spikes_widths
sterr_widths = stdev_widths / np.sqrt(spikes_widths.size)  #  standard error of the mean (SEM)


# x = np.linspace(-0.02, 0.1, 1000)
# y = st.norm(4.0, 2.0).pdf(x)
#

#   making Figures
fig, axes = plt.subplots(3, 1, figsize=(10, 6))
axes[0].plot(signals['Vm'])
axes[0].plot(peaks, signals['Vm'][peaks], "*")
axes[0].set_xlabel("samples")
axes[0].set_ylabel("voltage (mV)")
axes[0].hlines(*width_half[1:], color="C2")  # zoom single spike, you will see half width line
#axes[0].hlines(*width_full[1:], color="C3")

mn, mx = plt.xlim(left=spike_intervals.min(), right=spike_intervals.max())
kde_xs = np.linspace(mn, mx, number_of_ISI+1)
kde = st.gaussian_kde(spike_intervals)
axes[1].hist(spike_intervals, density=True, bins=80)
axes[1].mn, mx = plt.xlim()
axes[1].set_xlim(mn, mx)
axes[1].kde_xs = np.linspace(mn, mx, number_of_ISI+1)
axes[1].kde = st.gaussian_kde(spike_intervals)
axes[1].plot(kde_xs, kde.pdf(kde_xs), label="PDF")
axes[1].set_xlabel("intervals (s)")
axes[1].set_ylabel("counts")

#         anotation in the figure #2
anotation = "CV = " + str(np.round(coef_var, 5))
anotation += "\n"
anotation += "firing rate = " + str(np.round(firing_rate, 5))
anotation += "\n"
anotation += "ISI mean = " + str(np.round(mean_of_ISI, 5)) + '\xB1' + str(np.round(stdev_intervals, 5))
at = AnchoredText(anotation, prop=dict(size=10), frameon=True, loc='upper right')
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
axes[1].add_artist(at)


mi, ma = plt.xlim(left=spikes_widths.min(), right=spikes_widths.max())
kde_wh = np.linspace(mi, ma, number_of_all_widths)
gauss_kde = st.gaussian_kde(spikes_widths)
axes[2].hist(spikes_widths, density=True, bins=80)
axes[2].mi, ma = plt.xlim(left=spikes_widths.min(), right=spikes_widths.max())
axes[2].set_xlim(mi, ma)
axes[2].kde_wh = np.linspace(mi, ma, number_of_all_widths)
axes[2].gauss_kde = st.gaussian_kde(spikes_widths)
axes[2].plot(kde_wh, gauss_kde.pdf(kde_wh), label="PDF")
axes[2].set_xlabel("spike width (s)")
axes[2].set_ylabel("counts")

#         anotation in the figure #3
anotation = "CV = " + str(np.round(coef_var_width, 5))
anotation += "\n"
anotation += "widths mean = " + str(np.round(mean_of_spikes_widths, 5)) + '\xB1' + str(np.round(stdev_widths, 5))
at = AnchoredText(anotation, prop=dict(size=10), frameon=True, loc='upper right')
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
axes[2].add_artist(at)
