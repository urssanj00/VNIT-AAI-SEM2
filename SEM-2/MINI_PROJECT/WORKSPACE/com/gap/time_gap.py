import pandas as pd
import matplotlib.pyplot as plt

# Load your data
file_path = "path_to_your_data.csv"  # Update with the actual file path
df = pd.read_csv(file_path)

# Ensure the timestamp column is parsed as datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Sort data by timestamp (important for consistency)
df = df.sort_values(by='timestamp')

# Define the expected interval (5 minutes)
interval = '5T'

# Generate a complete time range from the minimum to the maximum timestamp
time_range = pd.date_range(start=df['timestamp'].min(), end=df['timestamp'].max(), freq=interval)

# Create a complete DataFrame with the expected timestamps
expected_df = pd.DataFrame({'timestamp': time_range})

# Merge the original data with the complete range to identify gaps
merged_df = pd.merge(expected_df, df, on='timestamp', how='left', indicator=True)

# Identify missing timestamps
missing_timestamps = merged_df[merged_df['_merge'] == 'left_only']['timestamp']

# Calculate the length of gaps (in terms of the number of intervals)
if len(missing_timestamps) > 0:
    missing_intervals = (missing_timestamps.diff().dt.total_seconds() / 60).fillna(5)
    missing_gaps = pd.DataFrame({'gap_start': missing_timestamps[:-1],
                                 'gap_end': missing_timestamps[1:],
                                 'gap_length_minutes': missing_intervals[1:]})
else:
    missing_gaps = pd.DataFrame(columns=['gap_start', 'gap_end', 'gap_length_minutes'])

# Output the results
print(f"Missing timestamps ({len(missing_timestamps)} gaps found):")
if len(missing_timestamps) > 0:
    print(missing_gaps)
else:
    print("No gaps detected!")

# Save missing timestamps and gaps to files for review (optional)
missing_timestamps.to_csv("missing_timestamps.csv", index=False)
missing_gaps.to_csv("missing_gaps.csv", index=False)

# Visualize the data with gaps
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'], df['value'], marker='o', label='Actual Data')
for i, gap in missing_gaps.iterrows():
    plt.axvspan(gap['gap_start'], gap['gap_end'], color='red', alpha=0.3, label='Gap' if i == 0 else None)
plt.xlabel("Timestamp")
plt.ylabel("Value")
plt.title("Data Feed with Gaps Highlighted")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Alert if any gap is longer than a threshold (e.g., 15 minutes)
threshold_minutes = 15
long_gaps = missing_gaps[missing_gaps['gap_length_minutes'] > threshold_minutes]
if len(long_gaps) > 0:
    print(f"\nALERT: Found {len(long_gaps)} gaps longer than {threshold_minutes} minutes!")
    print(long_gaps)
else:
    print("\nNo gaps exceed the threshold.")
