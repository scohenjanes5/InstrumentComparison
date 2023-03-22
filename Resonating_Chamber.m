% Set the path to the folder containing the audio files
folder_path="C:\Users\sande\Desktop\school\Senior Spring\Math and Music\InstrumentComparison\Recordings\Jupiter\Notes\";

%% Get all envselopes in a folder
close all
% Get a list of all audio files in the folder
audio_files = dir(fullfile(folder_path, '*.wav'));

% Loop through all audio files and plot their envelopes
figure;
hold on;
for i = 1:length(audio_files)
    % Load the audio file
    file_path = fullfile(folder_path, audio_files(i).name);
    [audio_data, fs] = audioread(file_path);

    % Get the envelope of the audio file
    X_log_mag_envelope = GetEnvelope(file_path);

    % Trim the envelope
    f = (0:length(X_log_mag_envelope)-1)*(fs/length(X_log_mag_envelope));
    f_cutoff = 25000;
    [~, idx_cutoff] = min(abs(f - f_cutoff));
    X_log_mag_envelope_trim = X_log_mag_envelope(1:idx_cutoff);

    % Plot the envelope
    plot(f(1:idx_cutoff), X_log_mag_envelope_trim, 'DisplayName', audio_files(i).name);
end

% Set plot title and axis labels
title('Spectral Envelopes of Audio Files');
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
legend('show', 'Location', 'northeast');




%% Get envelope from file
function [envelope] = GetEnvelope(filename)
    [audio_data, fs] = audioread(filename);
    
    % Calculate the FFT of the audio signal
    N = length(audio_data);
    X = fft(audio_data);
    
    % Calculate the log-magnitude spectrum
    X_log_mag = log(abs(X));
    
    % Compute the inverse FFT of the log-magnitude spectrum
    cepstrum = ifft(X_log_mag);
    
    % Apply a low-pass lifter to the cepstrum
    lifter_cutoff = 1500; % Hz
    lifter_length = round(fs/lifter_cutoff);
    lifter = [ones(1,lifter_length) zeros(1,N-2*lifter_length) ones(1,lifter_length)];
    cepstrum_liftered = cepstrum .* lifter';
    
    % Convert the liftered cepstrum back to the log-magnitude spectrum
    X_log_mag_envelope = real(fft(cepstrum_liftered));
    
    % Plot the original log-magnitude spectrum and the spectral envelope
    f = (0:N-1)*(fs/N);
    f_trim = f(f<=fs/2); % Trim the frequency axis
    X_log_mag_trim = X_log_mag(f<=fs/2); % Trim the log-magnitude spectrum
    envelope = X_log_mag_envelope(f<=fs/2); % Trim the spectral envelope
end
