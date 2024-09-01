import argparse
from src.config import DIRECTORY_PATH
from src.directory_selection import select_directories
from src.llm_integration import process_with_langchain
from src.file_processing import process_directory
from src.logging_utils import setup_logging

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process and enhance Obsidian notes.")
    parser.add_argument('--rag', type=str, help="Specify the directory or file to use with langchain.")
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()

    if args.rag:
        process_with_langchain(args.rag)
    else:
        selected_directories = select_directories(DIRECTORY_PATH)
        if not selected_directories:
            print("No directories selected. Exiting.")
            return

        for directory in selected_directories:
            process_directory(directory)

if __name__ == "__main__":
    main()
