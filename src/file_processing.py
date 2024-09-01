import os
import time
import json
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from tabulate import tabulate
from plyer import notification
from llm_integration import check_llm_connectivity, llm_communication
from logging_utils import setup_logging
from llm_integration import PromptProcessor
from config import OUTPUT_FOLDER, MAX_WORKERS, CHECKPOINT, VERBOSE, PROMPTS_CONFIG_FILE, attachments_transferred, plugins_transferred

def is_text_file(file_path):
    """Check if the file is a typical text file based on its extension."""
    return file_path.lower().endswith(('.txt', '.md'))

def read_files_in_directory(directory_path, min_file_size=10, max_file_size=20000):
    """Read and categorize text files based on size, excluding folders that begin with a dot."""
    eligible_files = []
    non_eligible_files = []
    print(f"Scanning directory: {directory_path}")  # Debugging output
    for root, dirs, files in os.walk(directory_path):
        # Skip directories that begin with a dot
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file_name in files:
            file_path = os.path.join(root, file_name)
            print(f"Found file: {file_path}")  # Debugging output
            if is_text_file(file_path):
                try:
                    file_size = os.path.getsize(file_path)
                    if min_file_size <= file_size <= max_file_size:
                        print(f"Eligible file: {file_path} (Size: {file_size})")  # Debugging output
                        eligible_files.append((file_path, file_size))
                    else:
                        print(f"Non-eligible file: {file_path} (Size: {file_size})")  # Debugging output
                        non_eligible_files.append((file_path, file_size))
                except OSError as e:
                    logging.error(f"Failed to get size for {file_path}: {str(e)}")
    return eligible_files, non_eligible_files


def read_file(file_path):
    """Read file content, ignoring undecodable characters."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            return file.read()
    except Exception as e:
        logging.error(f"Failed to read {file_path}: {str(e)}")
        return None

def save_processed_content(output_folder, relative_path, content):
    """Save processed content to the output folder, preserving the directory structure."""
    try:
        output_file_path = os.path.join(output_folder, relative_path)
        output_file_path = os.path.splitext(output_file_path)[0] + ".md"  # Ensure the output file has the .md extension
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)  # Create directories if they don't exist

        print(f"Attempting to save content to: {output_file_path}")  # Debugging output
        if not content:
            logging.error(f"No content to save for {relative_path}.")
            return

        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(content)
            print(f"Processed content successfully saved to: {output_file_path}")

    except OSError as e:
        logging.error(f"Failed to save processed content for {relative_path}: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error while saving content for {relative_path}: {str(e)}")






def log_processing_details(details, process_logger, total_time):
    """Log processing details to the process log file."""
    process_logger.info("\n" + tabulate(details, headers="keys", tablefmt="grid"))
    process_logger.info(f"\nTotal Processing Time: {total_time:.1f} seconds")

def display_summary(eligible_files, non_eligible_files, directory_path, output_directory):
    """Display a summary of the processing configuration and file status."""
    
    num_eligible_files = len(eligible_files)
    num_non_eligible_files = len(non_eligible_files)

    total_eligible_size = sum(file_size for _, file_size in eligible_files)
    total_non_eligible_size = sum(file_size for _, file_size in non_eligible_files)

    final_check_summary = {
        "Communication with LLM": "Yes" if llm_communication else "No",
        "Attachments Transferred": "Yes" if attachments_transferred else "No",
        "Plugins Transferred": "Yes" if plugins_transferred else "No"
    }

    print(f"\nProcessing Summary:")
    print(f"  - Number of eligible files: {num_eligible_files} ({total_eligible_size} bytes)")
    print(f"  - Number of non-eligible files: {num_non_eligible_files} ({total_non_eligible_size} bytes)")
    print(f"  - Using {MAX_WORKERS} threads for processing")
    print(f"  - Checkpoint enabled: {'Yes' if CHECKPOINT else 'No'}")
    print(f"  - Verbose mode: {'Enabled' if VERBOSE else 'Disabled'}")
    print(f"  - Directory being processed: {directory_path}")
    print(f"  - Output Directory: {output_directory}")
    print("\nFinal Check:")
    for check, status in final_check_summary.items():
        print(f"  - {check}: {status}")

def confirm_proceed():
    """Confirm whether to proceed with processing eligible files."""
    proceed = input("Press Enter to proceed with processing eligible files or 'n' to cancel: ").strip().lower()
    return proceed != 'n'

def process_file(processor, file_path, file_size, output_folder, process_logger, error_logger, root_directory):
    """Process a single file and handle errors."""
    try:
        # Calculate the relative path starting from the selected root directory (e.g., 'Interviews')
        relative_path = os.path.relpath(file_path, os.path.dirname(root_directory))
        
        if VERBOSE:
            print(f"\nProcessing file: {os.path.basename(file_path)}\n")
            print(f"Relative path for saving: {relative_path}")

        file_content = read_file(file_path)
        if file_content is None:
            return None

        start_time = time.time()
        final_content = processor.run(file_content, relative_path)
        processing_time = round(time.time() - start_time, 1)

        if final_content:  # Ensure there's content to save
            # print(f"Generated content for {file_path}: {final_content[:100]}...")  # Print first 100 chars for brevity
            print(f"Output file path: {os.path.join(output_folder, relative_path)}")
            save_processed_content(output_folder, relative_path, final_content)  # Save with preserved structure
        else:
            logging.warning(f"No content generated for {file_path}.")
            return None

        processing_details = {
            "File Name": os.path.basename(file_path),
            "Size (bytes)": file_size,
            "Processing Time (s)": f"{processing_time:.1f}"
        }

        if VERBOSE:
            print("\nFinal Content:\n", final_content)
        return processing_details
    except Exception as e:
        error_logger.error(f"Error processing file {file_path}: {e}")
        return None








def process_files_in_parallel(processor, eligible_files, output_folder, process_logger, error_logger, directory_path, max_workers=4):
    """Process eligible files using parallel processing."""
    processing_details = []
    error_occurred = False

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, processor, file_path, file_size, output_folder, process_logger, error_logger, directory_path): (file_path, file_size) for file_path, file_size in eligible_files}
        
        for future in as_completed(futures):
            result = future.result()
            if result:
                processing_details.append(result)
            else:
                error_occurred = True

    return processing_details, error_occurred

def process_files_with_progress(processor, eligible_files, output_folder, process_logger, error_logger, directory_path, max_workers=MAX_WORKERS):
    """Process eligible files with a progress bar, showing the current file."""
    processing_details = []
    error_occurred = False
    processed_files = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, processor, file_path, file_size, output_folder, process_logger, error_logger, directory_path): (file_path, file_size) for file_path, file_size in eligible_files}

        with tqdm(total=len(futures), desc="Processing Files") as pbar:
            for future in as_completed(futures):
                file_path, file_size = futures[future]
                pbar.set_description(f"Processing: {os.path.basename(file_path)}")
                result = future.result()
                if result:
                    processing_details.append(result)
                    processed_files.append(file_path)
                    save_checkpoint(processed_files)  # Save progress after each file
                else:
                    error_occurred = True
                pbar.update(1)

    return processing_details, error_occurred

def notify_completion(message):
    notification.notify(
        title='File Processing Completed',
        message=message,
        app_name='Markdown Processor',
        timeout=10
    )

def save_checkpoint(processed_files, checkpoint_file='checkpoint.json'):
    """Save the list of processed files to a checkpoint file."""
    try:
        with open(checkpoint_file, 'w') as f:
            json.dump(processed_files, f)
    except OSError as e:
        logging.error(f"Failed to save checkpoint: {str(e)}")

def load_checkpoint(checkpoint_file='checkpoint.json'):
    """Load the list of processed files from a checkpoint file."""
    try:
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logging.error(f"Failed to load checkpoint: {str(e)}")
    return []

def filter_unprocessed_files(eligible_files, processed_files):
    """Filter out files that have already been processed."""
    return [(file_path, file_size) for file_path, file_size in eligible_files if file_path not in processed_files]

def ask_yes_no(question):
    """Prompt the user with a yes/no question."""
    while True:
        answer = input(f"{question} (y/n): ").lower()
        if answer in ['y', 'n']:
            return answer == 'y'
        print("Please answer 'y' or 'n'.")

def copy_directory(src, dest, folder_type=None):
    """Copy a directory from src to dest, handling specific folder types like plugins or attachments."""
    global plugins_transferred, attachments_transferred
    try:
        if os.path.exists(src):
            shutil.copytree(src, dest, dirs_exist_ok=True)
            print(f"Copied {src} to {dest}")
            
            # Set flags based on the folder type
            if folder_type == "plugins":
                plugins_transferred = True
            elif folder_type == "attachments":
                attachments_transferred = True
        else:
            print(f"Source directory {src} does not exist.")
    except OSError as e:
        logging.error(f"Failed to copy directory {src} to {dest}: {str(e)}")


def read_attachment_folder(app_json_path):
    """Read the attachment folder path from the app.json file."""
    try:
        with open(app_json_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config.get("attachmentFolderPath", "Attachments")
    except (OSError, json.JSONDecodeError) as e:
        logging.error(f"Failed to read attachment folder from {app_json_path}: {str(e)}")
        return "Attachments"
    
def process_directory(directory):
    """Process all eligible files in the given directory."""
    check_llm_connectivity()  # Check LLM connectivity before processing

    process_logger, error_logger = setup_logging()

    eligible_files, non_eligible_files = read_files_in_directory(directory)
    
    if not eligible_files and not non_eligible_files:
        print(f"No suitable text files found in the directory: {directory}.")
        return
    
    processor = PromptProcessor(OUTPUT_FOLDER)
    processor.load_prompts_from_json(PROMPTS_CONFIG_FILE)

    processed_files = load_checkpoint() if CHECKPOINT else []
    eligible_files = filter_unprocessed_files(eligible_files, processed_files) if CHECKPOINT else eligible_files
    
    display_summary(eligible_files, non_eligible_files, directory, OUTPUT_FOLDER)

    if not confirm_proceed():
        print("Processing cancelled.")
        return

    total_start_time = time.time()

    processing_details, error_occurred = process_files_with_progress(
        processor, 
        eligible_files, 
        OUTPUT_FOLDER, 
        process_logger, 
        error_logger, 
        directory, 
        max_workers=MAX_WORKERS
    )

    total_processing_time = round(time.time() - total_start_time, 1)

    if processing_details:
        if VERBOSE:
            print("\nProcessing Summary:")
            print(tabulate(processing_details, headers="keys", tablefmt="grid"))
    else:
        print("No files were processed.")

    log_processing_details(processing_details, process_logger, total_processing_time)

    print(f"\nTotal Processing Time: {total_processing_time} seconds")

    if not error_occurred:
        notify_completion("Processing completed successfully!")
    else:
        notify_completion("Processing completed with errors. Please check the error log for details.")