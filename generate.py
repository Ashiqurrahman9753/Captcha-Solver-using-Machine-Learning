import os
import requests

# Define the maximum number of download retries
max_retries = 50

# Define the download directory
download_directory = "C:\Scalable\project_2\captchas"

# Ensure download directory exists
os.makedirs(download_directory, exist_ok=True)

# Path to the CSV file containing image URLs
csv_file_path = "C:\Scalable\project_2\habeebra-challenge-filenames.csv"

# Function to download image and handle errors with retries
def download_image_with_retry(url, file_path, max_retries):
    for _ in range(max_retries):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an exception for 4xx and 5xx status codes
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded: {url} as {file_path}")
            return True  # Return True if the download was successful
        except Exception as e:
            print(f"Failed to download {url}: {e}")
    return False  # Return False if all retries fail

# Read image URLs from the CSV file and download images with retries
with open(csv_file_path, "r") as file:
    for line in file:
        file_name = line.strip()
        image_url = f"https://cs7ns1.scss.tcd.ie/?shortname=habeebra&myfilename={file_name}"
        file_path = os.path.join(download_directory, file_name)
        if not download_image_with_retry(image_url, file_path, max_retries):
            print(f"Max retries reached for {image_url}. Skipping.")
