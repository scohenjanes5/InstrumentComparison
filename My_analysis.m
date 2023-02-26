%% Setup
chosenFolder="C:\Users\sande\Desktop\school\Senior Spring\Math and Music\InstrumentComparison\Recordings";
audioFiles = dir(fullfile(chosenFolder, '*.wav'));
ads = audioDatastore(fullfile(chosenFolder, {'Wonderphone/Notes', 'Jupiter/Notes'}), ...
    'IncludeSubfolders', true, 'FileExtensions', '.wav');
fileNames = ads.Files;

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

%% Check labels
countEachLabel(ads)

%% MFCCs
for i = 1:1%length(audioFiles)
    % Load the audio file
    audioFile = fullfile(chosenFolder, audioFiles(i).name);
    [audio, fs] = audioread(audioFile);
   
    % mfcc
    [coeffs1,delta,deltaDelta,loc] = mfcc(audio,fs);
    mfcc(audio,fs)

    win = hann(1024,"periodic");
    S = stft(audio,"Window",win,"OverlapLength",512,"Centered",false);

    coeffs = mfcc(S,fs,"LogEnergy","Ignore");


    % plot
    t = (0:size(audio,1)-1)/fs;
    sound(audio,fs)
    figure;
    plot(t,audio)
    xlabel("Time (s)")
    ylabel("Amplitude")
    title("Sample Utterance from Training Set")

    % plot MFCCs
    nbins = 60;
    coefficientToAnalyze = 4;

    figure;
    histogram(coeffs(:,coefficientToAnalyze+1),nbins,"Normalization","pdf")
    title(sprintf("Coefficient %d",coefficientToAnalyze))

    % get FT data
    [~,~,f_tpt,TPT_Fourier]=FTwav(audioFile);

    %plot Fourier transform
    figure;
    [~, filename, ~] = fileparts(audioFile);
    plot(f_tpt,abs(TPT_Fourier),'DisplayName',filename)
    xlim([0 7000])
    xlabel('Frequency (Hz)')
    ylabel('Magnitude')
    title('Fourier Transform')    
    legend
end


%% % Create an empty table to store MFCCs and category labels
% Create a table to store the category labels for each file
labelsTable = table('Size',[0,2],'VariableTypes',{'string','string'},'VariableNames',{'File','Category'});

% Loop through each file in the audioDatastore
while hasdata(ads)
    % Read in the next audio file
    [audio,audioInfo] = read(ads);
    fs = audioInfo.SampleRate;

    % Compute the MFCC coefficients for the audio file
    coeffs = mfcc(audio,fs);

    % Add the category label for this file to the labelsTable
    labelsTable = [labelsTable; table(string(ads.Files(ads.CurrentFileIndex)), "Wonderphone", 'VariableNames',{'File','Category'})];
end




%% Function to FT the contents of wav files
function [y, fs, fReturn, Fourier]=FTwav(fname)
    [y,fs] = audioread(fname);
    Fourier = fft(y); %This line performs a fourier transform on the flute sound file y-values
    Fourier=Fourier(1:end/2); %This line grabs the first half of the Fourier results
    fReturn = (0:length(y)-1)*fs/length(y); %This line generates a vector of frequencies the Fourier transform checks on
    fReturn = fReturn(1:end/2); %Again we grab only the first half for our plot.
end
