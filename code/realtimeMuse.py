import pandas as pd
import numpy as np  # Module that simplifies computations on matrices
import matplotlib.pyplot as plt  # Module used for plotting
from pylsl import StreamInlet, resolve_byprop
from realtimeFromFile import processFile  # Module to receive EEG data
import utils  # Our own utility functions


""" EXPERIMENTAL PARAMETERS """
# Modify these to change aspects of the signal processing

# Length of the EEG data buffer (in seconds)
# This buffer will hold last n seconds of data and be used for calculations
BUFFER_LENGTH = 10

# Length of the epochs used to compute the FFT (in seconds)
EPOCH_LENGTH = 2 #had to increase this so we had 

# Amount of overlap between two consecutive epochs (in seconds)
OVERLAP_LENGTH = 0.8

# Amount to 'shift' the start of each next consecutive epoch
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH

# Index of the channel(s) (electrodes) to be used
# 0 = left ear, 1 = left forehead, 2 = right forehead, 3 = right ear
INDEX_CHANNEL = [0]


def start():
    count = 0

    print("Looking for an EEG stream...")
    streams = resolve_byprop("type", "EEG", timeout=2)
    if len(streams) == 0:
        raise RuntimeError("Can't find EEG stream.")
    # Set active EEG stream to inlet and apply time correction
    print("Start acquiring data")
    inlet = StreamInlet(streams[0], max_chunklen=12)
    #eeg_time_correction = inlet.time_correction()

    # Get the stream info and description
    info = inlet.info()
    #description = info.desc()

    # Get the sampling frequency
    # This is an important value that represents how many EEG data points are
    # collected in a second. This influences our frequency band calculation.
    # for the Muse 2016, this should always be 256
    fs = int(info.nominal_srate())

    """ 2. INITIALIZE BUFFERS """

    # Initialize raw EEG data buffer
    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 6))
    filter_state = None  # for use with the notch filter

    # Compute the number of epochs in "buffer_length"
    n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) / SHIFT_LENGTH + 1))

    # Initialize the band power buffer (for plotting)
    # bands will be ordered: [T9, AF7, AF8, TP19]
    #band_buffer = np.zeros((n_win_test, 4))

    """ 3. GET DATA """

    # The try/except structure allows to quit the while loop by aborting the
    # script with <Ctrl-C>
    print("Press Ctrl-C in the console to break the while loop.")

    try:
        while True:
            # Obtain EEG data from the LSL stream
            eeg_data, timestamp = inlet.pull_chunk(
                timeout=1, max_samples=int(SHIFT_LENGTH * fs)
            )
            timestamps = np.array(timestamp)

            # Only keep the channels we're interested in
            ch_data = np.array(eeg_data)[:, 0:5]
            total_data = np.column_stack((timestamps, ch_data))
            # convert array into dataframe
            DF = pd.DataFrame(total_data)

            # save the dataframe as a csv file
            DF.to_csv("total.csv")

            eeg_buffer, filter_state = utils.update_buffer(
                eeg_buffer, total_data, notch=True, filter_state=filter_state
            )

            # Get newest samples from the buffer
            data_epoch = utils.get_last_data(eeg_buffer, EPOCH_LENGTH * fs)

            recording = pd.DataFrame(
                data=data_epoch,
                columns=["timestamps", "TP9", "AF7", "AF8", "TP10", "AUX Right"], 
            )
            
            DF = pd.DataFrame(recording)
            if (count > 2):  #initially the buffer holds just 0s so we need to wait to get it all out
                DF.to_csv("recording.csv", index=False)
                processFile("recording.csv") #get realtime prediction for this time period
            count = count + 1

    except KeyboardInterrupt:
        print("Closing!")



if __name__ == "__main__":
    start()
