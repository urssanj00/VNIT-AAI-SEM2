% Generalised program to compute discrete Fourier transforms.

% Collect input from the user
x = input('Enter the sequence: ');
n = input('Enter the length of the Fourier transform: ');

% Validate inputs
validateInputs(x, n);

% Append the sequence with zeros if required
x_padded = padSequence(x, n);

% DFT calculation
X = calculateDFT(x_padded, n);

% Print the results
printResults(X);

 
%% Function Definitions

% Ensure the inputs are valid
function validateInputs(sequence, length_n)
    if length_n < length(sequence)
        error('The Fourier transform length must be greater than or equal to the sequence length.');
    end
end

% Extend the sequence with zeros to match the required length
function padded_sequence = padSequence(sequence, length_n)
    padded_sequence = [sequence, zeros(1, length_n - length(sequence))];
end

% Calculate the Discrete Fourier Transform
function DFT_result = calculateDFT(sequence, length_n)
    DFT_result = zeros(1, length_n); % Initialize the result vector with zeros
    for k = 0:length_n-1
        for m = 0:length_n-1
            DFT_result(k+1) = DFT_result(k+1) + sequence(m+1) * exp(-1j * 2 * pi * k * m / length_n);
        end
    end
end

% Display the calculated DFT
function printResults(DFT_result)
    disp('Computed Discrete Fourier Transform:');
    disp(DFT_result);
end
