from collections import defaultdict
import pickle
import argparse
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd

from EEG_feature_extraction import (
    generate_feature_vectors_from_samples,
    matrix_from_csv_file,
)

place = 0


def init_plot():
    ani = FuncAnimation(plt.gcf(), plot_update, interval=100)  # update every 1 sec
    plt.show()


def plot_update(prediction):
    global place

    plt.cla()

    """
    concentrating = 0
    neutral = 1
    relaxed = 2
    """
    if prediction == 1:
        place -= 1
    if prediction == 2:
        place += 1
    plt.plot(place, 0, "ro")
    plt.xlim([-10, 10])
    plt.xticks(np.arange(-10, 10, 1))
    plt.yticks([])


def processFile(filename):
    FINAL_MATRIX = None
    full_file_path = filename

    currperiod = 0.0001  # small adjustment
    if "subject" in filename or "recording" in filename:
        currperiod = 1

    print("Processing ", full_file_path)
    chunk_size = 500
    #was 300
    for chunk in pd.read_csv(
        full_file_path,
        index_col=0,
        skipinitialspace=False,
        header=0,
        chunksize=chunk_size,
    ):
        # chunk is a DataFrame
        if len(chunk.index) >= chunk_size:
            pred(chunk, chunk_size, currperiod)


def pred(df, currcount, currperiod):
    df.to_csv(
        "testing.csv", encoding="utf-8"
    )  # made a new csv file just for this script
    filename = "testing.csv"
    model = pickle.load(open("xgb_model.sav", "rb"))  # using just the XBG classifier

    FINAL_MATRIX = None

    full_file_path = filename

    vectors, header = generate_feature_vectors_from_samples(
        file_path=full_file_path,
        nsamples=150,
        #was 150
        period=currperiod,
        state=0,
        remove_redundant=True,
        cols_to_ignore=-1,
        count=currcount,
    )
    print("resulting vector shape for the file", vectors.shape)

    if FINAL_MATRIX is None:
        FINAL_MATRIX = vectors
    else:
        FINAL_MATRIX = np.vstack([FINAL_MATRIX, vectors])

    print("FINAL_MATRIX", FINAL_MATRIX.shape)

    # Shuffle rows
    np.random.shuffle(FINAL_MATRIX)
    output_file = "pred.csv"

    np.savetxt(
        output_file, FINAL_MATRIX, delimiter=",", header=",".join(header), comments=""
    )

    df = pd.read_csv("../CognitiveLoad-FeatureGen/" + output_file)
    df = df.drop("Label", axis=1)
    pred = model.predict(df)
    proba = model.predict_proba(df)

    """classification notes:
        if state.lower() == 'concentrating':
            state = 0
        elif state.lower() == 'neutral':
            state = 1
        elif state.lower() == 'relaxed':
            state = 2
    """
    print(pred)
    print(proba)
    prediction = None
    # TODO: Incorporate the probability distribution of each classification
    # into the final classification

    # TODO: Find better bounds for these classifications
    avgclassification = np.mean(pred)
    if 0 <= avgclassification < 1:
        prediction = "stressed"
    elif 1 <= avgclassification < 2:
        prediction = "neutral"
    elif 2 <= avgclassification < 3:
        prediction = "relaxed"
    else:
        RuntimeError("Unable to determine a prediction")
    print("PREDICTION:", prediction)
    # plot_update(pred)


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename",
        type=str,
        help="Input the path to the file to process (relative to the root directory)",
    )
    FLAGS = parser.parse_args()
    processFile(filename=FLAGS.filename)
    # init_plot()
