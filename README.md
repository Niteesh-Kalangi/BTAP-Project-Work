In this repository the Cognitive Load subteam has developed our machine learning model for classifying the mental state of EEG data. Our work is based on Tim de Boer's model found [here](https://github.com/Timm877/BCI_Project) and Dr. Jordan Bird's feature generation work found [here](https://github.com/jordan-bird/eeg-feature-generation).

## Files included:
- EEG_generate_training_matrix.py: this script generates the features for the data in dataset/original_data and outputs them to out.csv. This is modified from Dr. Bird's repository and takes EEG brainwaves and creates a static dataset through a sliding window approach. 
- EEG_feature_extraction.py: this script goes through the process of generating features for each of the time slices in the above script. 
- XGBclassifier.py: this holds our XGBoost classifier that is trained and tested on the data in out.csv
- GRUclassify.py: this holds our GRU model that is trained and tested on the data in out.csv 
- realtimeFromFile.py: this script takes in a file as a parameter and then uses the XGB classifier and EEG_generate_training_matrix scripts to identify its features and perform a realtime prediction of the mental states in the file. Note, this script processes the file in small chunks of 300 rows.
- realtimeMuse.py: this script streams EEG data in realtime (using Petal Metrics and the Muselsl [library](https://github.com/alexandrebarachant/muse-lsl) and records the current buffer data in a file. Then, once about 5 seconds of data have been collected, it performs a realtime prediction on that file.

## To Do


