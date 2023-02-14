%% Setup
filePath = matlab.desktop.editor.getActiveFilename;
%Matlab is apparently dumber than python and won't change file separater on Windows to \ automatically
filepath_parts = split(filePath,'\');
folderPath = join(filepath_parts(1:end-1),'\');
cd(folderPath{1})
disp('Directory Set')
global duration;
duration=3;
global bitrate;
bitrate=16;
%global PlaybackID;
%global Recording_ID;

%% Pick directory to analyze
% Prompt the user to select a folder
chosenFolder = uigetdir('.', 'Select a folder');

% Display the chosen folder
disp(['Chosen folder: ' chosenFolder]);

%% Get the list of audio files in the chosen folder and do stuff to them.
audioFiles = dir(fullfile(chosenFolder, '*.wav'));

% Loop through each audio file
for i = 1:length(audioFiles)
    % Load the audio file
    audioFile = fullfile(chosenFolder, audioFiles(i).name);
    [audio, fs] = audioread(audioFile);
    
    %get FT data
    [~,~,f_tpt,TPT_Fourier]=FTwav(audioFile);

    % Extract just the filename (without extension) from the full path
    [~, filename, ~] = fileparts(audioFile);

    figure;
    plot(f_tpt,abs(TPT_Fourier),'DisplayName',filename)
    xlim([0 7000])
    legend
end

%% Function to FT the contents of wav files
function [y, fs, fReturn, Fourier]=FTwav(fname)
    [y,fs] = audioread(fname);
    Fourier = fft(y); %This line performs a fourier transform on the flute sound file y-values
    Fourier=Fourier(1:end/2); %This line grabs the first half of the Fourier results
    fReturn = (0:length(y)-1)*fs/length(y); %This line generates a vector of frequencies the Fourier transform checks on
    fReturn = fReturn(1:end/2); %Again we grab only the first half for our plot.
end
