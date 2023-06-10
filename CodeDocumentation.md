# Code Documentation
Contents:\
[Making Recordings](#making-recordings)\
[Preprocessing and training](#analysis-and-training)\
[Resonating Chamber Visualization](#resonating-chamber-characteristics)\
[Example trained network](#trained-network-example)\
[Batch visualize notes](#visualizing-notes)\
[Wav tone generator](#wav-generator)

## Making Recordings

### Setup

Recordings.m will check the audio input and output devices available and print their identifiers. Adjust the values of `PlaybackID` and `Recording_ID` to the values that correspond with the appropriate devices.

There is a cell that plays white noise through the chosen speaker and another that plots the waveform of a sound played into the input device. Verify the expected behavior is observed before collecting data.

### Data collection

Write the note name you would like to start at in the provided `record_notes()` function. After a countdown the three second recording will automatically start. The note will be played back on the selected playback device and a plot will also be created.

After the recording finishes, you will be prompted to approve the recording. `yes` will save the recording and begin the recording process for the next note in the chromatic sequence. Entering `abort` will quit the program without saving the file, `end` will quit the program with saving. Any other input will redo the recording.

The filenames will be related to their note names. It is noted that the octave numbers correspond with the concert pitches of the notes, but the note names are the Bb pitches, so some names have the wrong octave number. However, the names are still all unique. If anyone cares to PR I would welcome that though!

The files are all saved in the active directory, so they should be moved to a directory with the trumpet name to avoid overwriting them.

## Analysis and Training

My_analysis.m requires the folder path that contains subdirectories for each trumpet's notes. Labels will be inferred from these subdirectory names. Example MFCCs for one file will be plotted

The last cell was automatically generated by MATLAB. This trains the neural network and shows the results automatically.

## Resonating Chamber Characteristics

Ensure the paths in Resonating_Chamber.m are relevant. `folder_path` should contain the notes for one trumpet, and `root_folder` should be the directory that contains the subdirectories with trumpet names.

The envelopes for each note will be shown for the trumpet specified in `folder_path`, while only the averages will be shown for each trumpet in `root_folder`.

## Trained Network Example

Trained_Network.m was automatically generated by MATLAB. Because training takes only a few seconds, it is not necessary to use since a network trained to your specific data is preferred. This particular file was generated with fewer data files available, and is less accurate than rerunning the training on the data provided.

## Visualizing notes

TPT_FT.m will plot all FT graphs for the audio files in a given folder. This is currently not used in the project, but can be used to subjectively compare notes.

## Wav generator

WavMaker.m will make a wav file for a specific frequency and duration. This is useful for showing the utility of mels, as 165 -> 365 Hz appears much larger than 1560 -> 1760 Hz. 