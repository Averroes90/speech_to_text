import os
import ipywidgets as widgets
from IPython.display import display


def list_files_with_extensions(path, extensions):
    """List files in the given directory with specified extensions."""
    all_files = os.listdir(path)
    filtered_files = {
        os.path.basename(f): os.path.join(path, f)
        for f in all_files
        if any(f.endswith(ext) for ext in extensions)
    }
    return filtered_files


def create_file_selector(path, extensions):
    """Create a dropdown for selecting multiple files with specific extensions."""
    files = list_files_with_extensions(path, extensions)
    dropdown = widgets.SelectMultiple(
        options=sorted(files.keys()),  # Display file names
        description="Files:",
        disabled=False,
    )
    display(dropdown)
    return dropdown, files


def get_selected_file_paths(dropdown, file_map):
    return [file_map[file] for file in dropdown.value]


def change_file_extension(file_path, new_extension):
    # Split the file path into the root and the extension
    root, _ = os.path.splitext(file_path)

    # Ensure the new extension starts with a dot
    if not new_extension.startswith("."):
        new_extension = "." + new_extension

    # Combine the root with the new extension
    new_file_path = root + new_extension

    return new_file_path


def load_and_print_srt(file_path):
    try:
        # Open the SRT file in read mode
        with open(file_path, "r", encoding="utf-8") as file:
            # Read and print each line from the file
            for line in file:
                print(line.strip())  # strip() removes any leading/trailing whitespace
    except FileNotFoundError:
        print(f"The file at {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def extract_path_components(full_path):
    # Extract the directory path without the file name
    directory_path = os.path.dirname(full_path)

    # Extract the file name without the extension
    file_name_without_extension = os.path.splitext(os.path.basename(full_path))[0]

    # Extract the file extension
    extension = os.path.splitext(full_path)[1]

    return directory_path, file_name_without_extension, extension
