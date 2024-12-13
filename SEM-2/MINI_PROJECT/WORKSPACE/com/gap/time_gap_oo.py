import pandas as pd
import matplotlib.pyplot as plt

class TimeGapPM2Point5:
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
        print(self.df.head())

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

    def get_merged_df(self, expected_df):
        merged_df = pd.merge(expected_df, self.df, left_on=0, right_on=0, how='left', indicator=True)
        return merged_df
    # call 3

    def get_missing_timeline(self, df1):
        print("0. missing_timestamps")
        gap = []

        n = len(df1)
        i = 0

        while i < n:
            val = df1.iloc[i, 1]
            if pd.isnull(val):
                # Start of missing block
                if i > 0 and pd.notna(df1.iloc[i - 1, 1]):
                    gap.append(df1.iloc[i - 1, 0])
                while i < n and pd.isnull(df1.iloc[i, 1]):
                    i += 1
                if i < n and pd.notna(df1.iloc[i, 1]):
                    gap.append(df1.iloc[i, 0])
            i += 1

        df_gap = pd.DataFrame(gap, columns=["timestamp"])
        return df_gap

    def get_missing_timeline1(self, df1):
        print("0. missing_timestamps")
        #print(df1)
        gap = []
        for i in range(0, len(df1)):

            val = df1.iloc[i][1]
            if pd.isnull(val):
                #print(f'0 null for {df1.iloc[i][0]}')
                k = 0
                for j in range(i, len(df1)):
                    if k == 0:
                        st_val = df1.iloc[j - 1][0]
                        # if st_val not in gap:
                        # print(f'0 k={k} : adding for {st_val}')
                        if pd.notna(df1.iloc[j - 1][1]):
                            gap.append(st_val)
                        k = 1
                        j = j + 1
                    val1 = df1.iloc[j][1]
                    if pd.isnull(val1):
                        # print(f'1 k={k} : null for {df1.iloc[j][0]}')
                        continue
                        k = 1
                    else:
                        # print(f'1 k={k} : adding for {df1.iloc[j][0]}')
                        st_val = df1.iloc[j][0]
                        if st_val not in gap:
                            if pd.notna(df1.iloc[j][1]):
                                gap.append(st_val)
                        break
        #print("1. missing_timestamps")

        df_gap = pd.DataFrame(gap, columns=["timestamp"])
        return df_gap


    # call 3
    def get_gap_minutes(self, df_gap2):
        # Separate odd and even rows based on index
        odd_rows = df_gap2.iloc[::2].reset_index(drop=True)  # odd-indexed rows
        even_rows = df_gap2.iloc[1::2].reset_index(drop=True)  # even-indexed rows

        # Merge the odd and even rows side by side
        merged_df = pd.concat([odd_rows, even_rows], axis=1)
        merged_df.columns = ['timestamp_odd', 'timestamp_even']  # Rename columns

        #Calculate the time difference between the even and odd timestamps
        merged_df['time_diff'] = (merged_df['timestamp_even'] - merged_df[
           'timestamp_odd']).dt.total_seconds() / 60  # Time difference in minutes

        return merged_df

    # call 4
    def time_gap_to_csv(self, filepath1, filepath2, filepath3, missing_gaps, missing_timestamps, long_gap_alert_df):
        # Save missing timestamps and gaps to files for review (optional)
        missing_timestamps.to_csv(filepath1, index=False)
        missing_gaps.to_csv(filepath2, index=False)
        if long_gap_alert_df is not None:
            long_gap_alert_df.to_csv(filepath3, index=False)




    def plot_missing_timeline(self, merged_df, plot_file_path, data_file_name):

        df = merged_df
        # Plotting
        plt.figure(figsize=(12, 6))

        # Plot valid data points
        plt.plot(df[0][df['_merge'] == 'both'], df[1][df['_merge'] == 'both'],
                 label='Valid Data',
                 color='blue',
                 marker='o')

        # Highlight NaN values (actual missing values in column `1`)
        plt.scatter(
            df[0][df[1].isna()],
            [0] * df[1].isna().sum(),  # Use y=0 for NaN visibility
            label='Missing PM2.5 Concentration',
            color='orange',
            marker='s',
            zorder=5
        )

        # Highlight 'left_only' rows as explicit gaps
        plt.scatter(
            df[0][df['_merge'] == 'left_only'],
            [0] * (df['_merge'] == 'left_only').sum(),
            label="(Explicit Gap)",
            color='orange',
            marker='o',
            zorder=5
        )

        # Add labels and legend
        plt.title('Timestamp Gaps Analysis :'+data_file_name)
        plt.xlabel('Timestamp')
        plt.ylabel('PM 2.5 Concentration Value')
        plt.axhline(0, color='black', linewidth=0.5, linestyle='--')  # Reference line
        plt.legend()
        plt.grid(True)

        # Rotate x-axis labels for readability
        plt.xticks(rotation=75)
        plt.tight_layout()
        plt.savefig(plot_file_path )
        plt.close()


    # Alert if any gap is longer than a threshold (e.g., 15 minutes)
    def gap_alert(self, missing_gaps, threshold_minutes=15):
        print(f"{missing_gaps.columns}")
        print(f"{missing_gaps.head()}")
        long_gaps = missing_gaps[missing_gaps['time_diff'] > threshold_minutes]
        if len(long_gaps) > 0:
            print(f"long_gaps:")
            print(long_gaps.head())

            print(f"\nALERT: Found {len(long_gaps)} gaps longer than {threshold_minutes} minutes!")
            return long_gaps
        else:
            print("\nNo gaps exceed the threshold.")
            return None

# file_names = ["test-changed"]

file_names = ["Braamfischer", "Chamber_of_Mines", "Ikusasalethu_Sec", "iThemba_Labs_1", "iThemba_Labs_2",
             "J.B_Marks_Secondary", "Kya_Sand_site_1", "Kya_Sand_site_2", "Kya_Sand_site_3", "Kya_Sand_site_4",
             "Kya_Sand_site_5","Lufhereng_Sec", "Milpark_Netcare_1", "Milpark_NetCare_2", "Nkone_Maruping_Pry",
             "Nomzamo_Mandela_Primary_School", "Origin_Center_1", "Origin_Center_2", "Siyabonga_Secondary",
             "Slovoville", "Thulani", "Westdene"]

#file_names = ["Braamfischer"]


filext = ".csv"
file_dir = "../dataset/wits/"
output_dir = file_dir+"output/"
from pathlib import Path

gap_alert_cnt = 1
for file_name in file_names:
    op_folder_for_csv = output_dir+file_name
    op_folder_name = Path(op_folder_for_csv)

    # Create the folder
    op_folder_name.mkdir(exist_ok=True)
    print(f"Folder '{op_folder_name}' created successfully.")

    file_path = file_dir+file_name+filext
    output_file_path1 = op_folder_for_csv+"/"+file_name+"_missing_timestamps.csv"
    output_file_path2 = op_folder_for_csv+"/"+file_name+"_missing_gaps.csv"
    output_file_path3 = op_folder_for_csv+"/"+file_name+"_missing_gaps_alert.csv"

    plot_path = op_folder_for_csv+"/"+file_name+"_plot.png"

    # ../dataset/wits/test-changed.csv
    print(f'Starting processing for {file_path}')

    tg = TimeGapPM2Point5(file_path)

    time_range1 = tg.get_time_range()
    expected_df1 = tg.create_df_expected_timestamp(time_range1)
    merged_df1 = tg.get_merged_df(expected_df1)

    print("merged_df1")
    print(merged_df1.head())

    df_gap1 = tg.get_missing_timeline(merged_df1)
    print(f"df_gap1 : ")
    print(f"{df_gap1.head()}")
    if df_gap1 is not None:
        df_missing_timestamp = tg.get_gap_minutes(df_gap1)
        print(f"df_missing_timestamp : ")
        print(f"{df_missing_timestamp.head()}")
        # output_file_path1= ../dataset/wits/output/test-changed_missing_timestamps.csv
        # output_file_path2= ../dataset/wits/output/test-changed_missing_gaps.csv
        long_gap_alert_df = tg.gap_alert(df_missing_timestamp)
        # print('Gap Alerted')
        tg.time_gap_to_csv(output_file_path1, output_file_path2, output_file_path3, df_gap1, df_missing_timestamp,
                           long_gap_alert_df)

        print(f'Gap reports saved in CSV : ')
        print(f'1. {output_file_path1}')
        print(f'2. {output_file_path2}')
        print(f'3. {output_file_path3}')


        tg.plot_missing_timeline(merged_df1, plot_path, file_name+filext)
        if gap_alert_cnt == 1:
            print(f'Missing gaps plotted ')

        print(f'{gap_alert_cnt}.    {plot_path}')
        gap_alert_cnt = gap_alert_cnt+1




