import os
import re

def get_unique_path(filepath):
    directory, filename = os.path.split(filepath)
    name, ext = os.path.splitext(filename)

    max_counter = 0

    pattern = re.compile(rf"^{re.escape(name)}_(\d+){re.escape(ext)}$")

    for file in os.listdir(directory):
        match = pattern.match(file)
        if match:
            num = int(match.group(1))
            if num > max_counter:
                max_counter = num

    new_filepath = os.path.join(directory, f"{name}_{max_counter + 1}{ext}")

    return new_filepath