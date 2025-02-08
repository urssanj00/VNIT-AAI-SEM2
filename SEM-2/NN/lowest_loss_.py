import pandas as pd

# Load the CSV file
file_path = "assignment5-observation-reportv1.3.csv"
df = pd.read_csv(file_path)

# Sort the DataFrame by Loss in ascending order and Validation Accuracy % in descending order
df_sorted = df.sort_values(by=["Loss"], ascending=[True])

# Find the row with the lowest loss
lowest_loss_row = df_sorted.iloc[0]
lowest_loss_index = df_sorted.index[0]

# Print the results
print("Row with Lowest Loss (Index {}):".format(lowest_loss_index))
print(lowest_loss_row)



# Sort the DataFrame by Loss in ascending order and Validation Accuracy % in descending order
df_sorted = df.sort_values(by=["Loss", "Validation Accuracy %"], ascending=[True, False])

# Find the row with the lowest loss
lowest_loss_row = df_sorted.iloc[0]
lowest_loss_index = df_sorted.index[0]

# Find the row with the highest validation accuracy after sorting
highest_val_acc_row = df_sorted.iloc[df_sorted["Validation Accuracy %"].idxmax()]
highest_val_acc_index = df_sorted["Validation Accuracy %"].idxmax()

# Print the results
print("Row with Lowest Loss (Index {}):".format(lowest_loss_index))
print(lowest_loss_row)
print("\nRow with Highest Validation Accuracy (Index {}):".format(highest_val_acc_index))
print(highest_val_acc_row)