import pandas as pd
import matplotlib.pyplot as plt

# Load your data without column labels
file_path = "../dataset/wits/Braamfischer.csv"  # Update with the actual file path
df = pd.read_csv(file_path, header=None)

# Refer to the first and second columns by index
timestamp_column = df[0]  # First column (index 0) assumed to be timestamps
value_column = df[1]  # Second column (index 1) assumed to be values

# Ensure the timestamp column is parsed as datetime
timestamp_column = pd.to_datetime(timestamp_column)

# Sort data by the timestamp column (important for consistency)
df = df.sort_values(by=0)  # Sort by column 0 (timestamps)

# Define the expected interval (5 minutes)
interval = '5min'  # Use 'min' instead of 'T' to avoid deprecation warnings

# Generate a complete time range from the minimum to the maximum timestamp
time_range = pd.date_range(start=timestamp_column.min(), end=timestamp_column.max(), freq=interval)

# Create a complete DataFrame with the expected timestamps
expected_df = pd.DataFrame({0: time_range})  # Column 0 corresponds to timestamps in the generated range
expected_df[0] = pd.to_datetime(expected_df[0])
df[0] = pd.to_datetime(df[0])

print()
print(f'df.columns                          : expected_df.columns')
print(f'{df.columns}    {expected_df.columns}')
print()


# Merge the original data with the complete range to identify gaps
merged_df = pd.merge(expected_df, df, left_on=0, right_on=0, how='left', indicator=True)

# Identify missing timestamps
missing_timestamps = merged_df[merged_df['_merge'] == 'left_only'][0]  # Column 0 is the timestamp column

# Calculate the length of gaps (in terms of the number of intervals)
if len(missing_timestamps) > 0:
    missing_intervals = (missing_timestamps.diff().dt.total_seconds() / 60).fillna(5)
    missing_gaps = pd.DataFrame({
        'gap_start': missing_timestamps[:-1].values,
        'gap_end': missing_timestamps[1:].values,
        'gap_length_minutes': missing_intervals[1:].values
    })
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
plt.plot(timestamp_column, value_column, marker='o', label='Actual Data')
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
