A1 = 1;          % Amplitude
f1 = 50;         % Frequency in Hz

A2 = 1;        % Amplitude
f2 = 80;        % Frequency in Hz

fs = 1000;       % Sampling frequency (in Hz)
duration = 1;    % Duration of the signal in seconds

t = 0:1/fs:0.05; % Time vector from 0 to 0.05 seconds with the given sampling frequency
display(t)

% Generate the two sinusoidal signals
x1 = A1 * sin(2 * pi * f1 * t);   % First sine wave
x2 = A2 * sin(2 * pi * f2 * t);   % Second sine wave


combined_signal = x1 + x2;


% Adding random noise (Normal distribution noise with mean 0 and std dev 0.5)
noise = 0.5 * randn(size(t));  % Standard normal noise scaled by 0.5
noisy_signal = combined_signal + noise;


% Apply Fourier Transform to the noisy signal
N = length(noisy_signal); % Length of the signal
f = (0:N-1)*(fs/N);        % Frequency axis (0 to fs)

% Compute the Fourier Transform
Y = fft(noisy_signal);

% Compute the magnitude of the Fourier Transform
magnitude = abs(Y);

% Plot the magnitude response
figure;
plot(f, magnitude);
title('Magnitude Spectrum of the Noisy Signal');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
grid on;


% Analysis of the Spectrum:
% -  Noise will cause additional random peaks in the spectrum, reducing the
%    clarity of the signal's distinct frequencies.
% -  Peaks at 50 Hz and 80 Hz should still be prominent but may appear less 
%    sharp depending on the noise level.