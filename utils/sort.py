import os
import shutil
from datetime import datetime

# Use the current working directory as the source directory
source_directory = os.getcwd()

# Function to rename files and clean up "21kV" and trailing underscores
def clean_filename(filename):
    # Remove "21kV" if it exists
    new_filename = filename.replace("21kV", "")

    # Remove any trailing underscore before the .txt extension
    if new_filename.endswith("_.txt"):
        new_filename = new_filename.replace("_.txt", ".txt")

    # Remove double underscores if present
    new_filename = new_filename.replace("__", "_")
    
    # Only rename if the filename has changed
    if new_filename != filename:
        os.rename(os.path.join(source_directory, filename), os.path.join(source_directory, new_filename))
    
    return new_filename

# Function to extract date and time from filename
def extract_datetime_from_filename(filename):
    try:
        # Ensure the filename has the expected structure
        if '--' not in filename:
            raise ValueError(f"Filename '{filename}' does not contain expected delimiter '--'.")

        # Split the filename to extract the date and time part
        parts = filename.split("--")
        if len(parts) < 2:
            raise ValueError(f"Filename '{filename}' does not have the expected date/time part.")
        
        datetime_part = parts[1].split('_')[0]  # This ignores anything after the underscore

        # The format we're expecting: DayMonthDate_Time (e.g., "FriAug30_16h23")
        # Split based on "_" to separate date and time parts
        date_part, time_part = parts[1].split('_')

        # Ignore the portion after the underscore if there's more text (e.g., "21kV")
        time_part = time_part[:5]  # Ensure only the time is taken (e.g., "16h23")

        # Day is the first 3 characters, Month is the next 3 characters, Date is the remaining characters
        day = date_part[:3]  # "Fri"
        month = date_part[3:6]  # "Aug"
        date = date_part[6:]  # "30"

        # Combine these into a string that can be parsed by strptime
        datetime_str = f"{day} {month} {date}"

        # Create a datetime object; year is assumed as the current year
        dt = datetime.strptime(f"{datetime_str} {datetime.now().year}", "%a %b %d %Y")
        
        # Return the formatted date and time separately
        return dt.strftime("%Y-%m-%d"), time_part
    except Exception as e:
        print(f"Error processing file '{filename}': {e}")
        return None, None

# Get list of all files in the current directory
files = [f for f in os.listdir(source_directory) if os.path.isfile(os.path.join(source_directory, f))]

# First, clean filenames by removing "21kV" and any trailing underscores
files = [clean_filename(f) for f in files]

# Process each file
for file in files:
    date, time = extract_datetime_from_filename(file)
    
    # If the date is None, skip this file
    if date is None:
        continue
    
    # Create the directory structure: date/time
    date_directory = os.path.join(source_directory, date)
    time_directory = os.path.join(date_directory, time) if time else date_directory
    
    # Create directories if they don't exist
    if not os.path.exists(date_directory):
        os.makedirs(date_directory)
    if time and not os.path.exists(time_directory):
        os.makedirs(time_directory)
    
    # Move the file to the respective directory
    shutil.move(os.path.join(source_directory, file), os.path.join(time_directory, file))

print("Files have been cleaned, renamed, and organized into date and time directories.")
