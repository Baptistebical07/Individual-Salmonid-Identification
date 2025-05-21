import pandas as pd
from pathlib import Path

# Load the CSV
csv_path = r"path_for_dataset_file\to_be_modified.csv"  # Replace with actual CSV path to be modified (e.g. "where_the_file_is_stored\valid_metadata.csv")
df = pd.read_csv(csv_path)

# Define new base path
new_base_path = r"path_for_new_folder_direction"

# Update paths and ensure forward slashes
# Update paths, ensure forward slashes, and append .jpg if missing
def update_path(row):
    new_path = f"{new_base_path}\{row['class']}\{Path(row['path']).stem}".replace("\\", "/") # replacing the brackets is conditional of linux/windows/apple system
    if not new_path.endswith(".jpg"):
        new_path += ".jpg"
    return new_path

df["path"] = df.apply(update_path, axis=1)

# Save the updated CSV
output_csv_path = r"path_for_new_dataset_file\modified.csv"  # Replace with new output path  (e.g. "where_the_file_will_be_stored\new_valid_metadata.csv")
df.to_csv(output_csv_path, index=False)

print("CSV updated successfully!")