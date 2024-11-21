########################  eda_02_bike_sharing.py   ########################


from eda_01_bike_sharing import *
# Reading the Bike Dataset
bikesharing_df=pd.read_csv("..\\bikesharing_dataset\\hour.csv")
print(bikesharing_df.head())

bikesharing_df.info()
print(f"\nDataset Matrix Dimension:{bikesharing_df.shape}")