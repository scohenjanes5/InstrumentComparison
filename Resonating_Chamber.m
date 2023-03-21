% Load the audio file
file="C:\Users\sande\Desktop\school\Senior Spring\Math and Music\InstrumentComparison\Recordings\Jupiter\Notes\C5.wav";
[audio_data, fs] = audioread(file);

% Calculate the FFT of the audio signal
N = length(audio_data);
X = fft(audio_data);

% Calculate the log-magnitude spectrum
X_log_mag = log(abs(X));

% Compute the inverse FFT of the log-magnitude spectrum
cepstrum = ifft(X_log_mag);

% Apply a low-pass lifter to the cepstrum
lifter_cutoff = 1200; % Hz
lifter_length = round(fs/lifter_cutoff);
lifter = [ones(1,lifter_length) zeros(1,N-2*lifter_length) ones(1,lifter_length)];
cepstrum_liftered = cepstrum .* lifter';

% Convert the liftered cepstrum back to the log-magnitude spectrum
X_log_mag_envelope = real(fft(cepstrum_liftered));

% Plot the original log-magnitude spectrum and the spectral envelope
f = (0:N-1)*(fs/N);
f_trim = f(f<=fs/2); % Trim the frequency axis
X_log_mag_trim = X_log_mag(f<=fs/2); % Trim the log-magnitude spectrum
X_log_mag_envelope_trim = X_log_mag_envelope(f<=fs/2); % Trim the spectral envelope
figure;
plot(f_trim, X_log_mag_trim);
hold on;
plot(f_trim, X_log_mag_envelope_trim);
title('Log-Magnitude Spectrum and Spectral Envelope');
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
legend('Log-Magnitude Spectrum', 'Spectral Envelope');
