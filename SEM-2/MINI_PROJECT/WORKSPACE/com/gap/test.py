import pandas as pd
import matplotlib.pyplot as plt

'''
# Sample data
data = {
    0: [
        "2024-10-19 21:55:00+00:00", "2024-10-19 22:00:00+00:00", "2024-10-19 22:05:00+00:00",
        "2024-10-19 22:10:00+00:00", "2024-10-19 22:15:00+00:00", "2024-10-19 22:20:00+00:00",
        "2024-10-19 22:25:00+00:00", "2024-10-19 22:30:00+00:00", "2024-10-19 22:35:00+00:00",
        "2024-10-19 22:40:00+00:00", "2024-10-19 22:45:00+00:00", "2024-10-19 22:50:00+00:00",
        "2024-10-19 22:55:00+00:00", "2024-10-19 23:00:00+00:00", "2024-10-19 23:05:00+00:00",
        "2024-10-19 23:10:00+00:00", "2024-10-19 23:15:00+00:00", "2024-10-19 23:20:00+00:00",
        "2024-10-19 23:25:00+00:00", "2024-10-19 23:30:00+00:00", "2024-10-19 23:35:00+00:00",
        "2024-10-19 23:40:00+00:00", "2024-10-19 23:45:00+00:00", "2024-10-19 23:50:00+00:00",
        "2024-10-19 23:55:00+00:00"
    ],
    1: [
        7.68, 6.575, 7.03, 6.076667, None, 16.04, 18.005, 19.56, 19.09, None, 21.525, None,
        23.023333, 26.04, 24.575, None, 21.0, 20.065, 19.08, None, None, 26.07, 22.0, 19.585, 17.045
    ],
    '_merge': [
        'both', 'both', 'both', 'both', 'left_only', 'both', 'both', 'both', 'both', 'left_only',
        'both', 'left_only', 'both', 'both', 'both', 'left_only', 'both', 'both', 'both', 'left_only',
        'left_only', 'both', 'both', 'both', 'both'
    ]
}
'''
file_path = "../dataset/wits/Braamfischer.csv"
df = pd.read_csv(file_path, header=None)


# Convert to DataFrame
#df = pd.DataFrame(data)
df[0] = pd.to_datetime(df[0])  # Convert to datetime

df = df.sort_values(by=0)
print(df)
def csv_report_missing_timeline(df1):

    gap = []
    for i in range(0, len(df1)):

        start = pd.to_datetime( df1.iloc[i][0])
        val = df1.iloc[i][1]
        print (val)
        if val is None:
            print(f'0 null for {df1.iloc[i][0]}')
            k=0
            for j in range(i,len(df1)):
                if k == 0:
                    st_val = df1.iloc[j-1][0]
                    #if st_val not in gap:
                    #print(f'0 k={k} : adding for {st_val}')
                    if pd.notna(df1.iloc[j-1][1]):
                        gap.append(st_val)
                    k=1
                    j = j+1
                val1 = df1.iloc[j][1]
                if pd.isnull(val1):
                    #print(f'1 k={k} : null for {df1.iloc[j][0]}')
                    continue
                    k=1
                else:
                    #print(f'1 k={k} : adding for {df1.iloc[j][0]}')
                    st_val = df1.iloc[j][0]
                    if st_val not in gap:
                        if pd.notna(df1.iloc[j][1]):
                            gap.append(st_val)
                    break

    df_gap = pd.DataFrame(gap, columns=["timestamp"])
    return df_gap

df_gp = csv_report_missing_timeline(df)
print(df_gp)

def print_gap_minutes(df_gap):
    # Separate odd and even rows based on index
    odd_rows = df_gap.iloc[::2].reset_index(drop=True)  # odd-indexed rows
    even_rows = df_gap.iloc[1::2].reset_index(drop=True)  # even-indexed rows

    # Merge the odd and even rows side by side
    merged_df = pd.concat([odd_rows, even_rows], axis=1)
    merged_df.columns = ['timestamp_odd', 'timestamp_even']  # Rename columns

    # Calculate the time difference between the even and odd timestamps
    #merged_df['time_diff'] = (merged_df['timestamp_even'] - merged_df[
     #   'timestamp_odd']).dt.total_seconds() / 60  # Time difference in minutes

    print(merged_df)

df_gp = print_gap_minutes(df_gp)
print(df_gp)

def plot_missing_timeline():
    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot valid data points
    plt.plot(df[0][df['_merge'] == 'both'], df[1][df['_merge'] == 'both'], label='Valid Data', color='blue', marker='o')

    # Highlight NaN values (actual missing values in column `1`)
    plt.scatter(
        df[0][df[1].isna()],
        [0] * df[1].isna().sum(),  # Use y=0 for NaN visibility
        label='NaN (Missing Value)',
        color='blue',
        marker='s',
        zorder=5
    )

    # Highlight 'left_only' rows as explicit gaps
    plt.scatter(
        df[0][df['_merge'] == 'left_only'],
        [0] * (df['_merge'] == 'left_only').sum(),
        label="'left_only' (Explicit Gap)",
        color='orange',
        marker='o',
        zorder=5
    )

    # Add labels and legend
    plt.title('Data Gaps (NaN and left_only)')
    plt.xlabel('Timestamp')
    plt.ylabel('Value')
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')  # Reference line
    plt.legend()
    plt.grid(True)

    # Rotate x-axis labels for readability
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Show plot
    plt.show()


