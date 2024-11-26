import kagglehub

# Download latest version
path = kagglehub.dataset_download("sudalairajkumar/daily-temperature-of-major-cities")

print("Path to dataset files:", path)