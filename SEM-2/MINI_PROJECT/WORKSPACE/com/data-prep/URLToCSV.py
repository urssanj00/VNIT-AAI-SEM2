import requests
import csv
from datetime import datetime
import csv
import ijson
from PropertiesConfig import PropertiesConfig as PC
import pandas as pd

class URLToCSV:
    def __init__(self):
        # URL of the JSON data
        # Create an instance of PropertiesConfig
        properties_config = PC("sensor-data.properties")
        # Get properties as a dictionary
        self.properties = properties_config.get_properties_config()

        self.station_list_url = self.properties['station_list_url']
        self.data_set_dir = self.properties['data_set_path']
        # Generate a timestamp for the filename
        timestamp = datetime.now().strftime(self.properties['time_format'])
        current_date = datetime.now().strftime(self.properties['date_format'])
        self.station_list_csv_file = f"{self.data_set_dir}/station_list_{current_date}.csv"
        self.station_list_df = None
        self.station_url_list = []

    def dump_csv(self, url, file_path):
        #file_path = self.station_list_csv_file

        # Stream JSON from URL
        response = requests.get(url, stream=True)

        if response.status_code == 200:
            # Open the CSV file for writing
            with open(file_path, mode='w', newline='', encoding='utf-8') as file:
                writer = None

                # Stream parse the JSON data
                for record in ijson.items(response.raw, 'item'):
                    # Initialize CSV writer and write header once
                    if writer is None:
                        writer = csv.DictWriter(file, fieldnames=record.keys())
                        writer.writeheader()

                    # Write each record
                    writer.writerow(record)

            print(f"Data has been written to {file_path}")
        else:
            print(f"Failed to fetch data. HTTP Status Code: {response.status_code}")

    def m1_dump_station_data_list(self):
        self.dump_csv(self.station_list_url, self.station_list_csv_file)

    def m2_read_station_list_in_df(self):
        # Read the CSV file
        self.station_list_df = pd.read_csv(self.station_list_csv_file)

        # Display the first few rows of the dataframe
        print(self.station_list_df.head())

    def m3_build_station_url_list(self):
        st_id_df = self.station_list_df['_id']

        for st_id in st_id_df:
            print(st_id)
            self.station_url_list.append(f"{self.station_list_url}/{st_id}/sensorData?days=")

    def m4_dump_station_pm2_5_data(self, days):
        i=0
        for st_id in self.station_url_list:
            url_st_data = st_id + str(days)
            filename = f'{self.data_set_dir}/{i}.csv'
            print(filename)
            self.dump_csv(url_st_data, filename)
            i = i+1

urlToCSV = URLToCSV()

urlToCSV.m1_dump_station_data_list()
urlToCSV.m2_read_station_list_in_df()

urlToCSV.m3_build_station_url_list()
urlToCSV.m4_dump_station_pm2_5_data(1)

print(f"{urlToCSV.station_url_list}")
print("test")
