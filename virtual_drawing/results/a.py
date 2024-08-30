import os

# Loop to create folders from 1 to 100
for i in range(1, 101):
    folder_name = str(i)  # Convert the number to a string to use as a folder name
    os.makedirs(folder_name, exist_ok=True)  # Create the folder, ignore if it already exists

print("Folders 1 to 100 created successfully.")
