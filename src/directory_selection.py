import os

def get_subdirectories(directory_path):
    subdirectories = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
    return subdirectories

def select_directories(directory_path):
    subdirectories = get_subdirectories(directory_path)
    if not subdirectories:
        print("No subdirectories found.")
        return []

    print("Subdirectories found:")
    for i, subdir in enumerate(subdirectories, 1):
        print(f"{i}. {subdir}")

    print("Enter the number(s) of the directories you want to process, separated by commas (or 'a' for all):")
    selection = input("> ").strip().lower()

    if selection == 'a':
        return [os.path.join(directory_path, subdir) for subdir in subdirectories]
    else:
        selected_indices = [int(i.strip()) - 1 for i in selection.split(',')]
        return [os.path.join(directory_path, subdirectories[i]) for i in selected_indices if i in range(len(subdirectories))]
