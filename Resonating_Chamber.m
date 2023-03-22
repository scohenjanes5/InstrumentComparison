% Set the path to the folder containing the audio files
folder_path="C:\Users\sande\Desktop\school\Senior Spring\Math and Music\InstrumentComparison\Recordings\Jupiter\Notes\";
%%
plotEnvs(folder_path, 25000);
%%
[envMatrix, avgEnv] = plotEnvs2(folder_path, 25000);

%%
[f_trim, envelope_trim] = get_Env("C:\Users\sande\Desktop\school\Senior Spring\Math and Music\InstrumentComparison\Recordings\Jupiter\Notes\C5.wav", 2500);
[f_trim2, envelope_trim2] = get_Env("C:\Users\sande\Desktop\school\Senior Spring\Math and Music\InstrumentComparison\Recordings\Jupiter\Notes\C4.wav", 2500);

envMatrix = [envelope_trim; envelope_trim2];

avgEnv = mean(envMatrix,1);

disp(avgEnv)

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

%% Get all envselopes in a folder
function plotEnvs(folder_path, f_cutoff)
    close all
    % Get a list of all audio files in the folder
    audio_files = dir(fullfile(folder_path, '*.wav'));
    
    % Loop through all audio files and plot their envelopes
    figure;
    hold on;
    for i = 1:length(audio_files)
        % Load the audio file
        file_path = fullfile(folder_path, audio_files(i).name);
        [~, fs] = audioread(file_path);
        
        % Get the envelope of the audio file
        envelope = GetEnvelope(file_path);
        
        % Trim the envelope
        f = (0:length(envelope)-1)*(fs/length(envelope));
        [~, idx_cutoff] = min(abs(f - f_cutoff));
        envelope_trim = envelope(1:idx_cutoff);
        
        % Plot the envelope
        f=f(1:idx_cutoff);
        plot(f, envelope_trim, 'DisplayName', audio_files(i).name);
    end
    
    % Set plot title and axis labels
    title('Spectral Envelopes of Audio Files');
    xlabel('Frequency (Hz)');
    ylabel('Magnitude (dB)');
    legend('show', 'Location', 'northeast');
end



%% Get all envselopes in a folder
function [f_trim, envelope_trim] = get_Env(file_path, f_cutoff)
    [~, fs] = audioread(file_path);
        
    % Get the envelope of the audio file
    envelope = GetEnvelope(file_path);
    
    % Trim the envelope
    f = (0:length(envelope)-1)*(fs/length(envelope));
    [~, idx_cutoff] = min(abs(f - f_cutoff));
    
    f_trim=f(1:idx_cutoff);
    envelope_trim = envelope(1:idx_cutoff);
end

%%
function [envMatrix, avgEnv] = plotEnvs2(folder_path, f_cutoff)
    close all
    % Get a list of all audio files in the folder
    audio_files = dir(fullfile(folder_path, '*.wav'));
    
    % Initialize envelope matrix
    envMatrix = [];

    % Loop through all audio files and plot their envelopes
    figure;
    hold on;
    for i = 1:length(audio_files)
        % Load the audio file
        file_path = fullfile(folder_path, audio_files(i).name);
        [f_trim, envelope_trim] = get_Env(file_path, f_cutoff);
       
        % Add envelope to envelope matrix
        envMatrix = [envMatrix; envelope_trim];
        plot(f_trim, envelope_trim, 'DisplayName', audio_files(i).name);
    end
    
    % Compute average envelope
    avgEnv = mean(envMatrix, 1);
    

    % Plot individual envelopes
    title('Spectral Envelopes of Audio Files');
    xlabel('Frequency (Hz)');
    ylabel('Magnitude (dB)');
    legend('show', 'Location', 'northeast');
    
    % Plot average envelope
    figure;
    plot(f_trim, avgEnv, 'DisplayName', 'Average Envelope');
    title('Average Spectral Envelope of Audio Files');
    xlabel('Frequency (Hz)');
    ylabel('Magnitude (dB)');
    legend('show', 'Location', 'northeast');
end

