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

    def m1_generate_station_list_csv(self):
        self.station_list_df = self.get_response(self.station_list_url)
        print('m1_generate_station_list_csv')
        self.station_list_df.to_csv(f'{self.station_list_csv_file}', index=False, sep=',', encoding='utf-8')

    def m2_generate_pm2p5_csv(self, days):
        count = 1
        for st_id, city, name in zip(self.station_list_df["_id"], self.station_list_df["city"], self.station_list_df["name"]):
            pm2p5_df = self.add_lat_long(st_id, days)
            if pm2p5_df is not None:
                fname_temp = f'{city}_{name}_{count}'
                fname = fname_temp.replace(" ", "_")
                filename = f'{self.data_set_dir}/{fname}.csv'

                pm2p5_df.to_csv(filename, index=False, sep=',', encoding='utf-8')
                print(f'm2_generate_pm2p5_csv : written {filename}')
                count = count + 1

    def get_response(self, url):
        # Fetch the data from the URL
        response = requests.get(url)
        df = None
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON data into a Python object
            data = response.json()
            #print("Fetched JSON Data:", data)
            if not data:
                print("Empty JSON Data:")
            else:
                # Convert the dictionary to a DataFrame
                df = pd.json_normalize(data)
        else:
            print(f"Failed to fetch data. HTTP Status Code: {response.status_code}")
        return df

    def add_lat_long(self, station_id, days):
        url_sensor_details = f'{self.station_list_url}/{station_id}'
        url_pm2p5_data = f'{url_sensor_details}/sensorData?days={days}'

        sensor_details_df = self.get_response(url_sensor_details)
        sensor_details_df_exploded = sensor_details_df.explode('sensorIds', ignore_index=True)
        #print(sensor_details_df_exploded.head())
        print(sensor_details_df_exploded.columns)

        selected_columns_df = sensor_details_df_exploded[['sensorIds', 'longitude', 'latitude', 'province', 'city',
                                                          'name']]

        data_with_location1 = None
        pm2p5_df = self.get_response(url_pm2p5_data)
        if pm2p5_df is not None:
            # Merge with different column names for joining
            data_with_location = pm2p5_df.merge(selected_columns_df, left_on='sensor_id', right_on='sensorIds',
                                                how='left')

            # Check if the 'sensorIds' column exists before trying to drop it
            if 'sensorIds' in data_with_location.columns:
                data_with_location1 = data_with_location.drop('sensorIds', axis=1)
            else:
                print("'sensorIds' column not found in merged DataFrame")


        return data_with_location1
        #data_with_location1.to_csv('output.csv', index=False, sep=',', encoding='utf-8')


urlToCSV = URLToCSV()
urlToCSV.m1_generate_station_list_csv()
urlToCSV.m2_generate_pm2p5_csv(1)

