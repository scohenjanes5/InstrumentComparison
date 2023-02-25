projectDir = "C:\Users\sande\Desktop\school\Senior Spring\Math and Music\InstrumentComparison\Recordings";

%% Create audio Datastore object
ads = audioDatastore(fullfile(projectDir, {'Wonderphone/Notes', 'Jupiter/Notes'}), ...
    'IncludeSubfolders', true, 'FileExtensions', '.wav');
fileNames = ads.Files;

%TPT_IDs = extractBetween(fileNames,"Recordings\","Notes");
%ads.Labels = categorical(TPT_IDs);

% Extract labels from filenames and add them to the datastore
for i = 1:numel(ads.Files)
    [~, filename, ~] = fileparts(ads.Files{i});
    parentdir = extractAfter(fileparts(ads.Files{i}), 'Recordings\');
    parentdir = extractBefore(parentdir, '\');
    labels{i} = parentdir;
end

% Convert labels to categorical array
uniqueLabels = unique(labels);
categoricalLabels = categorical(labels, uniqueLabels);

% Assign categorical labels to datastore
ads.Labels = categoricalLabels;

countEachLabel(ads)

%% Split data into sets
% Split the datastore into training, evaluation, and test sets
[adsTrain, adsEval, adsTest] = splitEachLabel(ads, 0.6, 0.2, 0.2, 'randomized');

%% Example file
[audio,audioInfo] = read(adsTrain);
fs = audioInfo.SampleRate;

t = (0:size(audio,1)-1)/fs;
sound(audio,fs)
plot(t,audio)
xlabel("Time (s)")
ylabel("Amplitude")
axis([0 t(end) -1 1])
title("Sample Utterance from Training Set")
reset(adsTrain)


%% Setting Up Feature Extraction
numCoeffs = 20;
deltaWindowLength = 9;
windowDuration = 0.025;
hopDuration = 0.01;

windowSamples = round(windowDuration*fs);
hopSamples = round(hopDuration*fs);
overlapSamples = windowSamples - hopSamples;

afe = audioFeatureExtractor( ...
    SampleRate=fs, ...
    Window=hann(windowSamples,"periodic"), ...
    OverlapLength=overlapSamples, ...
    ...
    mfcc=true, ...
    mfccDelta=true, ...
    mfccDeltaDelta=true);
setExtractorParameters(afe,"mfcc",DeltaWindowLength=deltaWindowLength,NumCoeffs=numCoeffs)

%% Extracting Features
features = extract(afe,audio);
[numHops,numFeatures] = size(features)

