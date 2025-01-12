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

figure;
subplot(1, 1, 1);
stem(t, combined_signal, 'filled');
title('Combined Signal');
xlabel('Time(s)');
ylabel('Amplitude');
 