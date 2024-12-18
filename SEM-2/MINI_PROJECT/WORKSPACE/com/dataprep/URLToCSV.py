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
        i = 0
        for st_id, st_name in zip(self.station_url_list, self.station_list_df['name']):
            url_st_data = st_id + str(days)
            filename = f'{self.data_set_dir}/{st_name.lower().replace(" ", "_")}_{i}.csv'
            print(filename)
            self.dump_csv(url_st_data, filename)
            i = i + 1

    def get_response(self, url="https://try-again-test-isaiah.app.cern.ch/api/stations/66503397099ab1a7fbcfbd24"):
        # Fetch the data from the URL
        response = requests.get(url)
        df = None
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON data into a Python object
            data = response.json()
            print("Fetched JSON Data:", data)
            # Convert the dictionary to a DataFrame
            df = pd.json_normalize(data)
        else:
            print(f"Failed to fetch data. HTTP Status Code: {response.status_code}")
        return df

    def add_lat_long(self):
        url_pm2p5_data = "https://try-again-test-isaiah.app.cern.ch/api/stations/66503397099ab1a7fbcfbd24/sensorData?days=1"
        url_sensor_details = "https://try-again-test-isaiah.app.cern.ch/api/stations/66503397099ab1a7fbcfbd24"

        sensor_details_df = self.get_response(url_sensor_details)
        sensor_details_df_exploded = sensor_details_df.explode('sensorIds', ignore_index=True)
        #print(sensor_details_df_exploded.head())
        print(sensor_details_df_exploded.columns)

        selected_columns_df = sensor_details_df_exploded[['sensorIds', 'longitude', 'latitude']]

        print(selected_columns_df.head())

        pm2p5_df = self.get_response(url_pm2p5_data)
        print(pm2p5_df.head())




        # Merge with different column names for joining
        data_with_location = pm2p5_df.merge(selected_columns_df, left_on='sensor_id', right_on='sensorIds',
                                            how='left')
        data_with_location1 = None
        # Check if the 'sensorIds' column exists before trying to drop it
        if 'sensorIds' in data_with_location.columns:
            data_with_location1 = data_with_location.drop('sensorIds', axis=1)
        else:
            print("'sensorIds' column not found in merged DataFrame")

        print(data_with_location1)
        data_with_location1.to_csv('output.csv', index=False, sep=',', encoding='utf-8')


urlToCSV = URLToCSV()
urlToCSV.add_lat_long()
'''
urlToCSV.m1_dump_station_data_list()
urlToCSV.m2_read_station_list_in_df()

urlToCSV.m3_build_station_url_list()
urlToCSV.m4_dump_station_pm2_5_data(1)

print(f"{urlToCSV.station_url_list}")
print("test")
'''