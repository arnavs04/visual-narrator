import os

def get_top_5_largest_files(directory):
    # List to store the file sizes and names
    files_with_sizes = []

    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                file_size = os.path.getsize(file_path)
                files_with_sizes.append((file_size, file_path))
            except OSError as e:
                print(f"Error getting size of {file_path}: {e}")

    # Sort the files by size in descending order
    files_with_sizes.sort(reverse=True, key=lambda x: x[0])

    # Get the top 5 largest files
    top_5_files = files_with_sizes[:5]

    # Print the results
    for size, file in top_5_files:
        size_mb = size / (1024 * 1024)  # Convert bytes to MB
        print(f"File: {file}, Size: {size_mb:.2f} MB")

get_top_5_largest_files('visual-narrator/data/flickr8k')