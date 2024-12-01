import pandas as pd

# Define the missing timestamps
missing_timestamps = [
    "2024-10-19T23:55:00.000Z",
    "2024-10-19T23:50:00.000Z",
    "2024-10-19T23:45:00.000Z",
    "2024-10-19T23:40:00.000Z",
    "2024-10-19T23:25:00.000Z",
    "2024-10-19T23:20:00.000Z",
    "2024-10-19T23:15:00.000Z",
    "2024-10-19T23:05:00.000Z",
    "2024-10-19T23:00:00.000Z",
    "2024-10-19T22:55:00.000Z",
    "2024-10-19T22:45:00.000Z",
    "2024-10-19T22:35:00.000Z",
    "2024-10-19T22:30:00.000Z",
    "2024-10-19T22:25:00.000Z",
    "2024-10-19T22:20:00.000Z",
    "2024-10-19T22:10:00.000Z",
    "2024-10-19T22:05:00.000Z",
    "2024-10-19T22:00:00.000Z",
    "2024-10-19T21:55:00.000Z"
]

# Convert to pandas datetime series
timestamps = pd.to_datetime(missing_timestamps)

# Initialize an empty list to store expanded intervals
all_intervals = []

# Iterate over pairs of consecutive timestamps
for start, end in zip(timestamps[:-1], timestamps[1:]):
    # Create a range of 5-minute intervals between start and end
    range_between = pd.date_range(start=start, end=end, freq="5min")
    all_intervals.extend(range_between[:-1])  # Exclude the end timestamp manually

# Add the last timestamp to include it as well
all_intervals.append(timestamps[-1])  # Use standard indexing

# Convert back to a pandas series for display
all_intervals = pd.Series(all_intervals)

# Display the filled intervals
print(all_intervals)
