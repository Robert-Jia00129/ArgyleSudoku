import os
import matplotlib.pyplot as plt


def read_time_from_line(file_path, line_num):
    """Read a specific line from a file and return the time value."""
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            if i == line_num:
                return float(line.split(',')[0])
    return None


def find_matching_file(base_file, files, target_constraint):
    """Find the file name that matches the base file except for a specific constraint."""
    base_parts = base_file.split('-')
    for file in files:
        file_parts = file.split('-')
        if len(base_parts) == len(file_parts) and all(
                bp == fp or bp in target_constraint for bp, fp in zip(base_parts, file_parts)):
            return file
    return None


def plot_comparison_for_constraint_extended_range(files_directory, constraint_a='distinct', constraint_b='PbEq',
                                                  time_cap=200):
    all_files = os.listdir(files_directory)
    x_times_full = []
    y_times_full = []
    x_times_holes = []
    y_times_holes = []

    for file_name in all_files:
        if constraint_a in file_name or constraint_b in file_name:
            matching_file = find_matching_file(file_name.replace(constraint_a, constraint_b), all_files,
                                               [constraint_a, constraint_b])
            if matching_file:
                base_file_path = os.path.join(files_directory, file_name)
                matching_file_path = os.path.join(files_directory, matching_file)

                base_lines = sum(1 for line in open(base_file_path))
                match_lines = sum(1 for line in open(matching_file_path))
                if base_lines != match_lines:
                    print(f"Inconsistent line numbers: {file_name} and {matching_file}")

                min_lines = min(base_lines, match_lines)
                time_type = 'full' if 'full_time' in file_name else 'holes'

                with open(base_file_path, 'r') as base_file, open(matching_file_path, 'r') as match_file:
                    for i, (base_line, match_line) in enumerate(zip(base_file, match_file)):
                        if i >= min_lines:
                            break
                        base_time = min(float(base_line.split(',')[0]), time_cap)
                        matching_time = min(float(match_line.split(',')[0]), time_cap)

                        if time_type == 'full':
                            x_times_full.append(base_time)
                            y_times_full.append(matching_time)
                        else:
                            x_times_holes.append(base_time)
                            y_times_holes.append(matching_time)

    plt.figure(figsize=(10, 6))
    if x_times_full and y_times_full:
        plt.scatter(x_times_full, y_times_full, color='green', alpha=0.5, label='Full Time')
    if x_times_holes and y_times_holes:
        plt.scatter(x_times_holes, y_times_holes, color='blue', alpha=0.5, label='Holes Time')
    plt.plot([0, time_cap], [0, time_cap], 'r--')  # Line y=x for reference
    plt.xlim(0, time_cap)
    plt.ylim(0, time_cap)
    plt.xlabel(f'Times for {constraint_a} capped at {time_cap} seconds')
    plt.ylabel(f'Times for {constraint_b} capped at {time_cap} seconds')
    plt.title(f'Comparison of Times: {constraint_a} vs. {constraint_b} within [0, {time_cap}] seconds')
    plt.legend()
    plt.grid(True)
    plt.show()



def plot_comparison_for_constraint_limited_range(files_directory, constraint_a='distinct', constraint_b='PbEq',
                                                 time_cap=25):
    all_files = os.listdir(files_directory)
    x_times_full = []
    y_times_full = []
    x_times_holes = []
    y_times_holes = []

    for file_name in all_files:
        if constraint_a in file_name or constraint_b in file_name:
            matching_file = find_matching_file(file_name.replace(constraint_a, constraint_b), all_files,
                                               [constraint_a, constraint_b])
            if matching_file:
                base_file_path = os.path.join(files_directory, file_name)
                matching_file_path = os.path.join(files_directory, matching_file)

                base_lines = sum(1 for line in open(base_file_path))
                match_lines = sum(1 for line in open(matching_file_path))
                if base_lines != match_lines:
                    print(f"Inconsistent line numbers: {file_name} and {matching_file}")

                min_lines = min(base_lines, match_lines)
                time_type = 'full' if 'full_time' in file_name else 'holes'

                with open(base_file_path, 'r') as base_file, open(matching_file_path, 'r') as match_file:
                    for i, (base_line, match_line) in enumerate(zip(base_file, match_file)):
                        if i >= min_lines:
                            break
                        base_time = min(float(base_line.split(',')[0]), time_cap)
                        matching_time = min(float(match_line.split(',')[0]), time_cap)

                        if time_type == 'full':
                            x_times_full.append(base_time)
                            y_times_full.append(matching_time)
                        else:
                            x_times_holes.append(base_time)
                            y_times_holes.append(matching_time)

    plt.figure(figsize=(10, 6))
    if x_times_full and y_times_full:
        plt.scatter(x_times_full, y_times_full, color='green', alpha=0.5, label='Full Time')
    if x_times_holes and y_times_holes:
        plt.scatter(x_times_holes, y_times_holes, color='blue', alpha=0.5, label='Holes Time')
    plt.plot([0, time_cap], [0, time_cap], 'r--')  # Line y=x for reference
    plt.xlim(0, time_cap)
    plt.ylim(0, time_cap)
    plt.xlabel(f'Times for {constraint_a} capped at {time_cap} seconds')
    plt.ylabel(f'Times for {constraint_b} capped at {time_cap} seconds')
    plt.title(f'Comparison of Times: {constraint_a} vs. {constraint_b} within [0, {time_cap}] seconds')
    plt.legend()
    plt.grid(True)
    plt.show()



#  TODO: Haven't implemented cases where there are time outs but time less than 200, but would
#   affect the general picture tho
files_directory = '/Users/jiazhenghao/Desktop/CodingProjects/ArgyleSudoku/time-record'
plot_comparison_for_constraint_extended_range(files_directory, 'distinct', 'PbEq')
plot_comparison_for_constraint_limited_range(files_directory, 'distinct', 'PbEq')