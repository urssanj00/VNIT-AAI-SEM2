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



% Apply Fourier Transform to the combined signal
N = length(combined_signal); % Length of the signal
f = (0:N-1)*(fs/N);        % Frequency axis (0 to fs)

% Compute the Fourier Transform
Y = fft(combined_signal);

% Compute the magnitude of the Fourier Transform
magnitude = abs(Y);

% Plot the magnitude response
figure;
plot(f, magnitude);
title('Magnitude Spectrum of the Combined Signal');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
grid on;


% Analysis- Fourier Transform: 
% The FFT will show 
% peaks at the frequencies 
% of the sinusoidal components f1 and f2, 
% which should appear as spikes in 
% the magnitude response. The noise 
% will contribute to higher-frequency 
% components, manifesting as 
% additional small peaks in the spectrum.