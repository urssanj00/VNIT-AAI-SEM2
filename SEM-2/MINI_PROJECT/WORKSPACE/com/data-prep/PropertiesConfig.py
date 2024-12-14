from jproperties import Properties

class PropertiesConfig:
    def __init__(self, properties_file_name):
        configs = Properties()
        with open(properties_file_name, 'rb') as read_prop:
            configs.load(read_prop)
        self.propertiesConfig = configs.items()
        #print(type(self.propertiesConfig))

    def get_properties_config(self):
        properties_dict = {}
        for item in self.propertiesConfig:
            key, prop_tuple = item
            properties_dict[key] = prop_tuple.data
            #print(f"Key: {key}, Value: {prop_tuple.data}")
        return properties_dict

# Create an instance of PropertiesConfig
propertiesConfig = PropertiesConfig("sensor-data.properties")
# Get properties as a dictionary
properties = propertiesConfig.get_properties_config()

st = properties['station_list_url']
pt = properties['data_set_path']
print(f'station_list_url:{st}')
print(f'data_set_path:{pt}')