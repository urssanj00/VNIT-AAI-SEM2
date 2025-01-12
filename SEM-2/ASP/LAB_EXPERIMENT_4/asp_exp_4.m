% Task (c): Plot original signal, impulse trains, and sampled signals

% Original signal
f = 5;                          % Frequency of the signal (Hz)
t = linspace(-1, 1, 1000);      % Time vector
a = cos(2 * pi * f * t);        % Define the Original Signal

% Sampling frequencies
fs1 = 8;                        % i: fs < 2f
fs2 = 10;                       % ii: fs = 2f
fs3 = 50;                       % iii: fs >> 2f

% Sampling intervals
Ts1 = 1 / fs1;
Ts2 = 1 / fs2;
Ts3 = 1 / fs3;

% Sampled time vectors
t1 = -1:Ts1:1;
t2 = -1:Ts2:1;
t3 = -1:Ts3:1;

% Sampled signals
a1 = cos(2 * pi * f * t1);
a2 = cos(2 * pi * f * t2);
a3 = cos(2 * pi * f * t3);

% Plotting

% i. Plot the original signal
figure(1);
plot(t, a, 'b', 'LineWidth', 1.5);
title('Original Signal: a = cos(2\pi f t)');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;
legend('Original Signal');


% ii. Plot impulse trains for three sampling frequencies
figure(2);
stem(t1, ones(size(t1)), 'r', 'LineWidth', 1.5, 'Marker', 'none'); hold on;
stem(t2, 2 * ones(size(t2)), 'g', 'LineWidth', 1.5, 'Marker', 'none'); hold on;
stem(t3, 3 * ones(size(t3)), 'm', 'LineWidth', 1.5, 'Marker', 'none');
title('Impulse Trains for Different Sampling Frequencies');
xlabel('Time (s)');
ylabel('Amplitude');
legend('fs < 2f', 'fs = 2f', 'fs >> 2f');
grid on;

figure(3);
subplot(3, 1, 1);
stem(t1, a1, 'r', 'LineWidth', 1.5); hold on;
plot(t, a, 'b--'); % Overlay original signal
title('Sampled Signal: i. (fs < 2f)');
xlabel('Time (s)');
ylabel('Amplitude');
legend('Sampled Signal', 'Original Signal');
grid on;

subplot(3, 1, 2);
stem(t2, a2, 'g', 'LineWidth', 1.5); hold on;
plot(t, a, 'b--'); % Overlay original signal
title('Sampled Signal: ii. (fs = 2f)');
xlabel('Time (s)');
ylabel('Amplitude');
legend('Sampled Signal', 'Original Signal');
grid on;

subplot(3, 1, 3);
stem(t3, a3, 'm', 'LineWidth', 1.5); hold on;
plot(t, a, 'b--'); % Overlay original signal
title('Sampled Signal: iii. (fs >> 2f)');
xlabel('Time (s)');
ylabel('Amplitude');
legend('Sampled Signal', 'Original Signal');
grid on;


% d) Reconstruct the signal from the three sampled signals using the sinc interpolation technique:
% i.    Plot each reconstructed signal along with the original signal in a single plot. Comment
%       on undersampling, oversampling, and aliasing effects, and provide appropriate titles
%       and axis labels for each plot.

% sinc function 
function sinc_val = sinc_val(x)
    sinc_val = (sin(pi * x) ./ (pi * x));   % Element-wise division
    sinc_val(x == 0) = 1;                   % case when x = 0
end


% Reconstruct the signals using sinc interpolation
reconstructed_a1 = zeros(size(t));
reconstructed_a2 = zeros(size(t));
reconstructed_a3 = zeros(size(t));

for k = 1:length(t1)
    reconstructed_a1 = reconstructed_a1 + a1(k) * sinc_val(fs1 * (t - t1(k)));
end

for k = 1:length(t2)
    reconstructed_a2 = reconstructed_a2 + a2(k) * sinc_val(fs2 * (t - t2(k)));
end

for k = 1:length(t3)
    reconstructed_a3 = reconstructed_a3 + a3(k) * sinc_val(fs3 * (t - t3(k)));
end


% Plot the reconstructed signals along with the original signal
figure;

% Case i: fs < 2f
subplot(3, 1, 1);
plot(t, a, 'b', 'LineWidth', 1.5); hold on;
plot(t, reconstructed_a1, 'r--', 'LineWidth', 1.2);
title('Reconstructed Signal: Case i (fs < 2f)');
xlabel('Time (s)');
ylabel('Amplitude');
legend('Original Signal', 'Reconstructed Signal');
grid on;

% Case ii: fs = 2f
subplot(3, 1, 2);
plot(t, a, 'b', 'LineWidth', 1.5); hold on;
plot(t, reconstructed_a2, 'g--', 'LineWidth', 1.2);
title('Reconstructed Signal: Case ii (fs = 2f)');
xlabel('Time (s)');
ylabel('Amplitude');
legend('Original Signal', 'Reconstructed Signal');
grid on;

% Case iii: fs >> 2f
subplot(3, 1, 3);
plot(t, a, 'b', 'LineWidth', 1.5); hold on;
plot(t, reconstructed_a3, 'm--', 'LineWidth', 1.2);
title('Reconstructed Signal: Case iii (fs >> 2f)');
xlabel('Time (s)');
ylabel('Amplitude');
legend('Original Signal', 'Reconstructed Signal');
grid on;

% 2. Generate a frequency spectrum plot for:
% (a) The original signal.
% (b) The sampled signal at the three different frequencies.
% (c) The reconstructed signal.


% FFT for the reconstructed signals
A1_rec = fftshift(fft(reconstructed_a1));   % FFT of reconstructed signal at fs1
n1 = length(A1_rec);                        % Frequency axis for the FFT
f1 = linspace(-fs1/2, fs1/2, n1);           % Frequency vectors

A2_rec = fftshift(fft(reconstructed_a2));   % FFT of reconstructed signal at fs2
n2 = length(A2_rec);                        % Frequency axis for the FFT
f2 = linspace(-fs2/2, fs2/2, n2);           % Frequency vectors

A3_rec = fftshift(fft(reconstructed_a3));   % FFT of reconstructed signal at fs3
n3 = length(A3_rec);                        % Frequency axis for the FFT
f3 = linspace(-fs3/2, fs3/2, n3);           % Frequency vectors


% 2. Generate a frequency spectrum plot for:
% (a) original signal spectrum
figure;
subplot(4,1,1);
plot(t, abs(a)); % Magnitude of FFT
title('Frequency Spectrum of Original Signal');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
grid on;

% (b) The sampled signal at the three different frequencies.
% Plot spectrum of sampled signal at fs1
subplot(4,1,2);
plot(t1, abs(a1)); % Magnitude of FFT
title('Frequency Spectrum of Sampled Signal (fs = 8 Hz)');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
grid on;

% Plot spectrum of sampled signal at fs2
subplot(4,1,3);
plot(t2, abs(a2)); % Magnitude of FFT
title('Frequency Spectrum of Sampled Signal (fs = 10 Hz)');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
grid on;

% Plot spectrum of sampled signal at fs3
subplot(4,1,4);
plot(t3, abs(a3)); % Magnitude of FFT
title('Frequency Spectrum of Sampled Signal (fs = 50 Hz)');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
grid on;

% (c) The reconstructed signal.
figure;
% Plot spectrum of reconstructed signal at fs1
subplot(3,1,1);
plot(f1, abs(A1_rec)); % Magnitude of FFT
title('Frequency Spectrum of Reconstructed Signal (fs = 8 Hz)');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
grid on;

% Plot spectrum of reconstructed signal at fs2
subplot(3,1,2);
plot(f2, abs(A2_rec)); % Magnitude of FFT
title('Frequency Spectrum of Reconstructed Signal (fs = 10 Hz)');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
grid on;


% Plot spectrum of reconstructed signal at fs2
subplot(3,1,3);
plot(f3, abs(A3_rec)); % Magnitude of FFT
title('Frequency Spectrum of Reconstructed Signal (fs = 50 Hz)');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
grid on;