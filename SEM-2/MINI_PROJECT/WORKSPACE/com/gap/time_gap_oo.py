import pandas as pd
import matplotlib.pyplot as plt

class TimeGapPM2_5:
    #file_path = "../dataset/wits/Braamfischer.csv"  # Update with the actual file path
    def __init__(self, f_path, interval='5min'):
        self.df = pd.read_csv(f_path, header=None)

        # Refer to the first and second columns by index
        self.timestamp_column = self.df[0]  # First column (index 0) assumed to be timestamps
        self.value_column = self.df[1]  # Second column (index 1) assumed to be values

        # Ensure the timestamp column is parsed as datetime
        self.timestamp_column = pd.to_datetime(self.timestamp_column)

        # Sort data by the timestamp column (important for consistency)
        self.df = self.df.sort_values(by=0)  # Sort by column 0 (timestamps)
        print("Sorted dataset")
        print(self.df)

        # Define the expected interval (5 minutes)
        self.interval = interval  # Use 'min' instead of 'T' to avoid deprecation warnings

    # call 0
    def get_time_range(self):
        # Generate a complete time range from the minimum to the maximum timestamp
        time_range = pd.date_range(start=self.timestamp_column.min(), end=self.timestamp_column.max(), freq=self.interval)
        return time_range

    # call 1
    def create_df_expected_timestamp(self, time_range):
        # Create a complete DataFrame with the expected timestamps
        expected_df = pd.DataFrame({0: time_range})  # Column 0 corresponds to timestamps in the generated range
        expected_df[0] = pd.to_datetime(expected_df[0])
        self.df[0] = pd.to_datetime(self.df[0])
        return expected_df

    # call 2
    def get_missing_timestamps(self, expected_df):
        # Merge the original data with the complete range to identify gaps
        merged_df = pd.merge(expected_df, self.df, left_on=0, right_on=0, how='left', indicator=True)
        print("Merged dataset")
        print(merged_df.sort_values(by=0))
        # Identify missing timestamps
        missing_timestamps = merged_df[merged_df['_merge'] == 'left_only'][0]  # Column 0 is the timestamp column
        print("missing_timestamps")
        print(missing_timestamps)
        return missing_timestamps

    # call 3
    def get_time_gap(self, missing_timestamps):
        # Calculate the length of gaps (in terms of the number of intervals)
        if len(missing_timestamps) > 0:
            print(missing_timestamps.diff().dt.total_seconds() / 60)
            missing_intervals = (missing_timestamps.diff().dt.total_seconds() / 60).fillna(5)
            print("missing_intervals")
            print(missing_intervals)
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
            return missing_gaps
        else:
            print("No gaps detected!")
            return None

    # call 4
    def time_gap_to_csv(self, filepath1, filepath2, missing_gaps, missing_timestamps):
        # Save missing timestamps and gaps to files for review (optional)
        missing_timestamps.to_csv(filepath1, index=False)
        missing_gaps.to_csv(filepath2, index=False)

    # call 5
    def plot_missing_gaps(self, missing_gaps):
        # Visualize the data with gaps
        plt.figure(figsize=(12, 6))
        plt.plot(self.timestamp_column, self.value_column, marker='o', label='Actual Data')
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
    def gap_alert(self, missing_gaps, threshold_minutes=15):
        long_gaps = missing_gaps[missing_gaps['gap_length_minutes'] > threshold_minutes]
        if len(long_gaps) > 0:
            print(f"\nALERT: Found {len(long_gaps)} gaps longer than {threshold_minutes} minutes!")
            print(long_gaps)
        else:
            print("\nNo gaps exceed the threshold.")

file_name = "test-changed"
filext = ".csv"
file_dir = "../dataset/wits/"
file_path = file_dir+file_name+filext
output_dir = file_dir+"output/"
output_file_path1 = output_dir+file_name+"_missing_timestamps.csv"
output_file_path2 = output_dir+file_name+"_missing_gaps.csv"


print(file_path)
print(output_file_path1)
print(output_file_path2)

# ../dataset/wits/test-changed.csv
tg = TimeGapPM2_5(file_path)

time_range1 = tg.get_time_range()
expected_df1 = tg.create_df_expected_timestamp(time_range1)
missing_timestamps1 = tg.get_missing_timestamps(expected_df1)
missing_gaps1 = tg.get_time_gap(missing_timestamps1)

if missing_gaps1 is not None:
    # output_file_path1= ../dataset/wits/output/test-changed_missing_timestamps.csv
    # output_file_path2= ../dataset/wits/output/test-changed_missing_gaps.csv
    tg.time_gap_to_csv(output_file_path1, output_file_path2, missing_gaps1, missing_timestamps1)
    print('Gap report saved in CSV')
    tg.gap_alert(missing_gaps1)
    print('Gap Alerted')
    tg.plot_missing_gaps(missing_gaps1)
    print('Missing gaps plotted')



