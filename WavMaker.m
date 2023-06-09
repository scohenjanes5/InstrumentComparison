makeWav(165, 1)
%% Generate wav files for arbitrary tones
% freq: Hz value, duration: seconds
function [] = makeWav(freq, duration)

    bitrate=44100;
    filename = strcat(int2str(freq), "Hz.wav");

    % Generate sinewave
    t = linspace(0, duration, duration*bitrate); % Time vector that corresponds to bitrate
    y = sin(2*pi*freq*t); % Formula for sinewave with given frequency

    audiowrite(filename, y, bitrate); % Save audio file with 44100 samples per second
end