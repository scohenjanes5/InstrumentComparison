% Set the path to the folder containing the audio files
folder_path="C:\Users\sande\Desktop\school\Senior Spring\Math and Music\InstrumentComparison\Recordings\Jupiter\Notes\";
example_file=fullfile(folder_path,"A#3.wav");
root_folder = 'C:\Users\sande\Desktop\school\Senior Spring\Math and Music\InstrumentComparison\Recordings\';

subfolders = {'Jupiter', 'Wonderphone', 'Conn5BNYS'};  % List of subfolders to include
folder_paths = {};  % Initialize the cell array of folder paths

for i = 1:length(subfolders)
    subfolder = subfolders{i};
    folder_path = fullfile(root_folder, subfolder, 'Notes');
    folder_paths = [folder_paths, folder_path];
end

%%
[avgEnv,f] = plotEnvs(folder_path, 1500);

%%
[f_trims, avg_envs] = plotAvgEnvs(folder_paths, 1500, "lin");

%%
%[mel_trims, mel_avg_envs] = plotAvgEnvs(folder_paths, 1500, "mel");
% Convert frequency to mel scale
f_trim = 2595 * log10(1 + f_trims / 700);
figure;
plot(f_trim', avg_envs');
title('Average Spectral Envelopes of Different Trumpets (Mel Scale)');
xlabel('Mel Frequency (mels)');
ylabel('Magnitude (dB)');
legend({'Jupiter', 'Wonderphone', 'Conn5BNYS'});

%%
[env, f_trims2] = GetEnvelope(example_file, 1500, "plot");

%% Get envelope from file
function [envelope, f_trim] = GetEnvelope(filename, lifter_cutoff, plotOrNo)
    [audio_data, fs] = audioread(filename);
    
    % Calculate the FFT of the audio signal
    N = length(audio_data);
    Y = fft(audio_data);
    
    % Calculate the log-magnitude spectrum
    Y_log_mag = log(abs(Y));
    
    % Compute the inverse FFT of the log-magnitude spectrum
    cepstrum = ifft(Y_log_mag);
    
    % Apply a low-pass lifter to the cepstrum
    lifter_length = round(fs/lifter_cutoff);
    lifter = [ones(1,lifter_length) zeros(1,N-2*lifter_length) ones(1,lifter_length)];
    cepstrum_liftered = cepstrum .* lifter';
    
    % Convert the liftered cepstrum back to the log-magnitude spectrum
    Y_log_mag_envelope = real(fft(cepstrum_liftered));
    
    % Plot the original log-magnitude spectrum and the spectral envelope
    f = (0:N-1)*(fs/N);
    f_trim = f(f<=fs/2); % Trim the frequency axis
    envelope = Y_log_mag_envelope(f<=fs/2); % Trim the spectral envelope

    if plotOrNo == "plot"
        figure;
        hold on;
        plot(f_trim, real(cepstrum(f<=fs/2)), 'DisplayName', 'Unfiltered Cepstrum');
        plot(f_trim, lifter(f<=fs/2), 'DisplayName', 'Lifter');
        plot(f_trim, real(cepstrum_liftered(f<=fs/2)), 'DisplayName', 'Filtered Cepstrum');
        hold off;
        title('Cepstral Liftering');
        xlabel('Quefrency (ms)');
        ylabel('Magnitude');
        legend('Location', 'best');
        hold off
    
        figure;
        plot(f_trim, Y_log_mag(f<=fs/2), 'DisplayName', 'Log-Magnitude Spectrum');
        title('Log Spectrum');
        xlabel('Frequency (Hz)');
        ylabel('Magnitude (dB)');
        legend('Location', 'best');
    end
end

%% Get Avg Envelope
function [avgEnv, f_trim] = getAvgEnv(folder_path, lifter_cutoff)
    close all
    % Get a list of all audio files in the folder
    audio_files = dir(fullfile(folder_path, '*.wav'));
    
    % Initialize envelope matrix
    envMatrix = [];

    % Loop through all audio files and plot their envelopes
    for i = 1:length(audio_files)
        % Load the audio file
        file_path = fullfile(folder_path, audio_files(i).name);
            
        % Get the envelope of the audio file
        [envelope, f] = GetEnvelope(file_path, lifter_cutoff, "no");
        
        f_trim=reshape(f(1:end), 1, []);
        envelope = reshape(envelope, 1, []);
        
        % Add envelope to envelope matrix
        envMatrix = [envMatrix; envelope];
    end
    
    % Compute average envelope
    avgEnv = mean(envMatrix, 1);
end


%% Plot Avg Envelope and components
function [avgEnv, f_trim] = plotEnvs(folder_path, lifter_cutoff)
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
            
        % Get the envelope of the audio file
        [ envelope, f ] = GetEnvelope(file_path, lifter_cutoff, "no");
        
        f_trim=reshape(f(1:end), 1, []);
        envelope = reshape(envelope, 1, []);
       
        % Add envelope to envelope matrix
        envMatrix = [envMatrix; envelope];
       plot(f_trim, envelope, 'DisplayName', audio_files(i).name);
    end

    name = extractAfter(folder_path, 'Recordings\');
    name = extractBefore(name, '\');
    
    % Compute average envelope
    avgEnv = mean(envMatrix, 1);
    
    % Plot average envelope with thicker line and darker shade
    plot(f_trim, avgEnv, 'DisplayName', 'Average Envelope', 'LineWidth', 3, 'Color', [0, 0, 0]);

    % Add title and labels
    title('Spectral Envelopes of Audio Files',name);
    xlabel('Frequency (Hz)');
    ylabel('Magnitude (dB)');
    
    % Add legend
    legend('show', 'Location', 'northeast');
    hold off;
end

function [f_trims, avg_envs] = plotAvgEnvs(folder_paths, f_cutoff, freqMode)
    % Initialize arrays to store f_trim and currAvgEnv for each folder
    f_trims = [];
    avg_envs = [];
    %names = {};

    % Loop through all folders and plot their average envelopes
    for i = 1:length(folder_paths)
        %disp(folder_paths{i})
        %parentdir = extractAfter(fileparts(folder_paths(i)), 'Recordings\');
        %disp(parentdir)

        % Get the average envelope and envelope matrix for the current folder
        [currAvgEnv, f_trim] = getAvgEnv(folder_paths{i}, f_cutoff);
        if freqMode == "mel"
            % Convert frequency to mel scale
            f_trim = 2595 * log10(1 + f_trim / 700);
        end
        f_trims = [f_trims; f_trim];
        avg_envs = [avg_envs; currAvgEnv];
        %names = [names; cellstr(parentdir)];
    end
    
    % Plot all average envelopes
    figure;
    plot(f_trims', avg_envs');
    
    % Add title and labels
    if freqMode == "mel"
        title('Average Spectral Envelopes of Different Trumpets (Mel Scale)');
        xlabel('Mel Frequency (mels)');
    else
        title('Average Spectral Envelopes of Different Trumpets (Linear Scale)');
        xlabel('Frequency (Hz)');
    end
    ylabel('Magnitude (dB)');
    legend({'Jupiter', 'Wonderphone', 'Conn5BNYS'});
end
