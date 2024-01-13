#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
## Version history:

2018:
	Original script by Dr. Luis Manso [lmanso], Aston University
	
2019, June:
	Revised, commented and updated by Dr. Felipe Campelo [fcampelo], Aston University
	(f.campelo@aston.ac.uk / fcampelo@gmail.com)
"""

import os
import sys
import numpy as np
from EEG_feature_extraction import generate_feature_vectors_from_samples


def gen_training_matrix(directory_path, output_file, cols_to_ignore):
    """
    Reads the csv files in directory_path and assembles the training matrix with 
    the features extracted using the functions from EEG_feature_extraction.

    Parameters:
            directory_path (str): directory containing the CSV files to process.
            output_file (str): filename for the output file.
            cols_to_ignore (list): list of columns to ignore from the CSV

Returns:
            numpy.ndarray: 2D matrix containing the data read from the CSV

    Author: 
            Original: [lmanso] 
            Updates and documentation: [fcampelo]
    """

    # Initialise return matrix
    FINAL_MATRIX = None

    for x in os.listdir(directory_path):

        # Ignore non-CSV files
        if not x.lower().endswith('.csv'):
            continue

        # For safety we'll ignore files containing the substring "test".
        # [Test files should not be in the dataset directory in the first place]
        if 'test' in x.lower():
            continue
        try:
            name, state, _ = x[:-4].split('-')
        except:
            print('Wrong file name', x)
            sys.exit(-1)
        if state.lower() == 'concentrating':
            state = 0
        elif state.lower() == 'neutral':
            state = 1
        elif state.lower() == 'relaxed':
            state = 2
        else:
            print('Wrong file name', x)
            sys.exit(-1)

        print('Using file', x)
        full_file_path = directory_path + '/' + x
        #first part contains subject -> bird's data, else it's ours
        #bird's data: period of 1 seconds
        #our data: period of 0.0001 seconds, don't have to multiply by 10 anymore
        currperiod = 0.0001 #small adjustment
        if ("subject" in x):
            currperiod = 1
        vectors, header = generate_feature_vectors_from_samples(file_path=full_file_path,
                                                                nsamples=150,
                                                                period=currperiod,
                                                                state=state,
                                                                remove_redundant=True,
                                                                cols_to_ignore=cols_to_ignore)

        print('resulting vector shape for the file', vectors.shape)

        if FINAL_MATRIX is None:
            FINAL_MATRIX = vectors
        else:
            FINAL_MATRIX = np.vstack([FINAL_MATRIX, vectors])

    print('FINAL_MATRIX', FINAL_MATRIX.shape)

    # Shuffle rows
    np.random.shuffle(FINAL_MATRIX)

    # Save to file
    np.savetxt(output_file, FINAL_MATRIX, delimiter=',',
               header=','.join(header),
               comments='')

    return None


directory_path = "../CognitiveLoad-FeatureGen/dataset/all_data"
output_file = "out.csv"
gen_training_matrix(directory_path, output_file, cols_to_ignore=-1)
