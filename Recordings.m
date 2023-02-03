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
global PlaybackID;
global Recording_ID;

%% What devices are available?
devices = audiodevinfo;
fprintf('List of available audio recording devices:\n');
for i = 1:length(devices.input)
    fprintf('%d: %s, ID %d\n',i,devices.input(i).Name,devices.input(i).ID);
end
fprintf('List of available audio playback devices:\n');
for i = 1:length(devices.output)
    fprintf('%d: %s, ID:%d\n',i,devices.output(i).Name,devices.output(i).ID);
end

%For playback, use %Speakers (Realtek(R) Audio) (Windows DirectSound) for future reference
%for recording use Microphone (USB Audio Device) (Windows DirectSound)

%% Select Devices
% Select a device for audio playback
PlaybackID = 5; % Example, replace with desired device index
%Speakers (Realtek(R) Audio) (Windows DirectSound) for future reference
Recording_ID=1;
%for recording use Microphone (USB Audio Device) (Windows DirectSound)

%% Speaker Test
audio = rand(44100,1); % Example audio data
%construct a player with all the info needed.
player = audioplayer(audio,44100,bitrate,PlaybackID);
% Play audio
play(player);

%% Mic Test
[audio,fs] = recordTPT();
player = audioplayer(audio,fs,bitrate,PlaybackID);
play(player)
PlotTPT(audio,fs)

%% Get Data
%F#3 is the lowest note on trumpet
record_notes("A#",5)

%% Saving audio data
function record_notes(start_note, start_octave)
    global bitrate;
    global PlaybackID;
    note_names = {'F#','G','G#','A','A#','B','C','C#','D','D#','E','F'};
    start_index = find(strcmp(note_names, start_note)); % starting note name
    octave = start_octave;
    for i = 1:length(note_names)*4
        current_index = mod(start_index + i - 1, 13);% cycle chromatically up
        current_note = note_names{current_index};
        if strcmp(current_note, 'D')
            octave = octave + 1;
        end
        filename = [current_note, num2str(octave), '.wav']; % include note name and octave in filename
        % Record the audio and save it to the file
        redo = true;
        while redo
            [y,fs]=recordTPT();
            
            player = audioplayer(y,fs,bitrate,PlaybackID);
            play(player);
            PlotTPT(y,fs);

            % Prompt to redo, save, or abort the recording
            disp(filename)
            redo_input = input('Redo, save, or abort the recording? (yes/no/end/abort): ', 's');
            if strcmpi(redo_input, 'abort')
                return;
            end
            if strcmpi(redo_input, 'end')
                audiowrite(filename, y, fs);
                return;
            end
            if ~strcmpi(redo_input, "yes")
                redo = false;
                audiowrite(filename, y, fs);
            end
        end
    end
end

%% Recording
function [y,fs]=recordTPT()
    global duration;
    global Recording_ID;
    global bitrate;
    recObj=audiorecorder(44100,bitrate,Recording_ID);
    disp("Begin singing in 3 seconds")
    pause(1)
    disp("2s")
    pause(1)
    disp("1s")
    pause(1)
    disp("now!")
    recordblocking(recObj,duration);
    disp("End singing.")
    y=getaudiodata(recObj);
    fs=recObj.SampleRate;
end

%% Plotting
function PlotTPT(audio_vector,fs)
    time=(1:length(audio_vector))/fs;  % Time vector on x-axis
    figure;
    plot(time,audio_vector);
    title("Trumpet Note Playback");
    xlabel('Time');
    ylabel('Amplitude');
end
