import os
import time
import glob
import shutil


def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def delete_files(del_file):
    # Find all files starting with 'simulation'
    files_to_delete = glob.glob(del_file)

    # Loop through and delete each file
    for file in files_to_delete:
        os.remove(file)


def check_recent_file(tpr_filename, time_limit=30):
    tpr_path = tpr_filename

    if not os.path.exists(tpr_path):
        return False

    # Get the current time and the last modification time of the file
    current_time = time.time()
    file_mod_time = os.path.getmtime(tpr_path)

    # Check if the file was generated or modified in the last 30 seconds
    if current_time - file_mod_time < time_limit:
        return True
    else:
        return False


def move_recent_items(destination):
    """
    Move all files and folders generated within the last 30 seconds in the current directory to the specified destination.

    :param destination: The path of the destination directory
    """
    # get the current time
    current_time = time.time()

    # get current directory
    current_dir = os.getcwd()
    os.makedirs(destination, exist_ok=True)

    for item in os.listdir(current_dir):
        item_path = os.path.join(current_dir, item)

        # Get the last modification time of the item
        mod_time = os.path.getmtime(item_path)

        # Check if the item was generated within the last 30 seconds
        if current_time - mod_time <= 5:
            dest_path = os.path.join(destination, item)

            try:
                # move item
                shutil.move(item_path, dest_path)
                # print(f"Item '{item}' has been moved to '{destination}'")
            except Exception as e:
                print(f"An error occurred while moving item '{item}': {e}")


# Function used to format a dictionary so that each element occupies one line
def format_dict(d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            yield '    ' * indent + str(key) + ':\n'
            yield from format_dict(value, indent + 1)
        else:
            yield '    ' * indent + str(key) + ': ' + str(value) + '\n'


