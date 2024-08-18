import os
import shutil
import logging
import json
import subprocess
import pyautogui
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from plyer import notification
from ollama import Client
import ollama
from tabulate import tabulate
from requests.exceptions import RequestException

# Configuration
MODEL = 'llama3'
TEMPERATURE = 0.0
MODEL_API_URL = 'http://localhost:11434'
MAX_FILE_SIZE = 20000
MIN_FILE_SIZE = 10
DIRECTORY_PATH = r'C:\Documents\Obsidian\My Vault'
OUTPUT_FOLDER = r'C:\Documents\Obsidian\Obsidian_Enhanced'
LOG_DIRECTORY = 'logs'
PROCESS_LOG_FILE = 'process_log.txt'
ERROR_LOG_FILE = 'process_errors_log.txt'
PROMPTS_CONFIG_FILE = 'prompts_config.json'
VERBOSE = False
MAX_WORKERS = min(8, multiprocessing.cpu_count())  # Adjust based on the number of CPU cores
CHECKPOINT = True
MAX_RETRIES = 3  # Maximum number of retries for API requests
RETRY_DELAY = 2  # Delay between retries in seconds

# Flags for Final Check
llm_communication = False
attachments_transferred = False
plugins_transferred = False


# Set up logging
def setup_logging(output_folder, log_directory='logs', process_log_file='process_log.txt', error_log_file='process_errors_log.txt'):
    """Setup logging configuration and prepare directories."""
    os.makedirs(log_directory, exist_ok=True)

    process_log_path = os.path.join(log_directory, process_log_file)
    error_log_path = os.path.join(log_directory, error_log_file)

    error_logger = logging.getLogger('error_logger')
    error_logger.setLevel(logging.ERROR)
    error_handler = logging.FileHandler(error_log_path)
    error_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    error_logger.addHandler(error_handler)

    process_logger = logging.getLogger('process_logger')
    process_logger.setLevel(logging.INFO)
    process_handler = logging.FileHandler(process_log_path)
    process_handler.setFormatter(logging.Formatter('%(message)s'))
    process_logger.addHandler(process_handler)

    return process_logger, error_logger

# File handling functions
def is_text_file(file_path):
    """Check if the file is a typical text file based on its extension."""
    return file_path.lower().endswith(('.txt', '.md'))

def read_files_in_directory(directory_path, min_file_size=10, max_file_size=20000):
    """Read and categorize text files based on size."""
    eligible_files = []
    non_eligible_files = []
    for root, _, files in os.walk(directory_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if is_text_file(file_path):
                try:
                    file_size = os.path.getsize(file_path)
                    if min_file_size <= file_size <= max_file_size:
                        eligible_files.append((file_path, file_size))
                    else:
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
    """Save processed content to the output folder."""
    try:
        output_file_path = os.path.join(output_folder, relative_path)
        output_file_path = os.path.splitext(output_file_path)[0] + ".md"
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(content)
    except OSError as e:
        logging.error(f"Failed to save processed content for {relative_path}: {str(e)}")

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

    # Final Check Summary
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

# Processing functions
class PromptProcessor:
    def __init__(self, output_folder, custom_hooks=None):
        self.model = MODEL
        self.temperature = TEMPERATURE
        self.output_folder = output_folder
        self.custom_hooks = custom_hooks if custom_hooks else []
        self.client = Client(host=MODEL_API_URL)  # Use the Ollama client

    def load_prompts_from_json(self, json_file):
        """Load prompts from a JSON file."""
        try:
            if VERBOSE:
                print("Loading prompts from JSON configuration...")
            with open(json_file, 'r', encoding='utf-8') as file:
                config = json.load(file)
                self.messages = config['messages']
        except (OSError, json.JSONDecodeError) as e:
            logging.error(f"Failed to load prompts from {json_file}: {str(e)}")

    def process_messages(self, content, file_name):
        """Send messages to the model and get the response."""
        messages = self._format_messages(content)

        try:
            response = self.client.chat(model=self.model, messages=messages)
            final_response = self._get_response_content(response)

            if not final_response:
                logging.error(f"No response generated for {file_name}.")
                return ""

            for hook in self.custom_hooks:
                final_response = hook(final_response)

            self._save_response(file_name, final_response)
            return final_response
        except ollama.ResponseError as e:
            logging.error(f"Failed to process messages for {file_name}: {e}")
            return ""

    def _format_messages(self, content):
        """Format messages with the provided content."""
        return [
            {
                "role": message['role'],
                "content": "\n".join([part['text'] for part in message['content']]).format(content=content)
            }
            for message in self.messages
        ]


    def _get_response_content(self, response):
        """Extract content from the model's response."""
        final_response = ""

        try:
            if isinstance(response, list):
                for part in response:
                    if isinstance(part, dict) and 'message' in part:
                        message_content = part['message'].get('content', '')
                        final_response += message_content
                        if VERBOSE and message_content:
                            print(message_content, end='', flush=True)
                    else:
                        logging.error(f"Unexpected response format: {part}")
            elif isinstance(response, dict):
                if 'message' in response:
                    final_response = response['message'].get('content', '')
                    if VERBOSE and final_response:
                        print(final_response, end='', flush=True)
                else:
                    logging.error(f"Unexpected response structure: {response}")
            else:
                logging.error(f"Unexpected response type: {type(response)}")

            if not final_response:
                logging.error("The final response content is empty.")

        except TypeError as e:
            logging.error(f"TypeError encountered while processing response: {e}")

        return final_response



    def _save_response(self, file_name, content):
        """Save the model's response to a file."""
        output_path = os.path.join(self.output_folder, file_name)
        output_path = os.path.splitext(output_path)[0] + ".md"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if not content:
            logging.error(f"No content to save for {file_name}.")
            return

        try:
            logging.info(f"Saving content to {output_path}...")
            with open(output_path, "w", encoding='utf-8') as f:
                f.write(content)
            logging.info(f"Content successfully saved to {output_path}.")
            if VERBOSE:
                print("\nFinal Content:\n", content)
        except OSError as e:
            logging.error(f"Failed to save response for {file_name}: {str(e)}")



    def run(self, initial_content, file_name):
        """Run the processor with the initial content and file name."""
        if VERBOSE:
            print(f"\n********************* Thinking...\n")
        return self.process_messages(initial_content, file_name)

def process_file(processor, file_path, file_size, output_folder, process_logger, error_logger, directory_path):
    """Process a single file and handle errors."""
    try:
        relative_path = os.path.relpath(file_path, directory_path)
        if VERBOSE:
            print(f"\nProcessing file: {os.path.basename(file_path)}\n")
        file_content = read_file(file_path)
        
        if file_content is None:
            return None

        start_time = time.time()
        final_content = processor.run(file_content, relative_path)
        processing_time = round(time.time() - start_time, 1)
        
        save_processed_content(output_folder, relative_path, final_content)

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

def open_obsidian_and_switch_to_graph(output_folder):
    """Open the Obsidian vault and switch to the graph view."""
    try:
        print("Opening Obsidian")
        print("DON'T TOUCH ANYTHING TILL YOU SEE THE GRAPH VIEW")
        obsidian_command = f'start "" "obsidian://open?vault={output_folder}"'
        subprocess.Popen(obsidian_command, shell=True)

        time.sleep(5)

        pyautogui.hotkey('enter')

        time.sleep(5)  # Adjust this delay based on your system's performance

        pyautogui.hotkey('ctrl', 'g')
    except Exception as e:
        logging.error(f"Failed to open Obsidian and switch to graph view: {str(e)}")

def setup_output_folder(output_folder):
    """Ensure the output folder exists and clear any existing contents."""
    try:
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder, exist_ok=True)
    except OSError as e:
        logging.error(f"Failed to setup output folder {output_folder}: {str(e)}")

def check_llm_connectivity():
    """Check if the Ollama LLM can be reached by attempting to load a simple model."""
    try:
        # Attempt to load a simple prompt or check the availability of the model
        client = Client(host=MODEL_API_URL)
        response = client.show(MODEL)

        if response:
            print("Ollama LLM is reachable.")
            return True
        else:
            print("Ollama LLM is not reachable.")
            return False
    except Exception as e:
        print(f"Failed to reach Ollama LLM: {str(e)}")
        return False

# Main function
def main():
    # Set up the output folder before copying any files or opening Obsidian
    setup_output_folder(OUTPUT_FOLDER)

    # Check if LLM is reachable
    global llm_communication
    llm_communication = check_llm_connectivity()

    if not llm_communication:
        print("Cannot reach LLM. Exiting the process.")
        return

    # Proceed with the rest of the processing...
    process_logger, error_logger = setup_logging(OUTPUT_FOLDER)
    
    processor = PromptProcessor(OUTPUT_FOLDER)
    processor.load_prompts_from_json(PROMPTS_CONFIG_FILE)

    processed_files = load_checkpoint() if CHECKPOINT else []

    eligible_files, non_eligible_files = read_files_in_directory(DIRECTORY_PATH)
    
    if not eligible_files and not non_eligible_files:
        print("No suitable text files found in the specified directory.")
        return

    eligible_files = filter_unprocessed_files(eligible_files, processed_files) if CHECKPOINT else eligible_files
    
    # Ask if the user wants to copy plugins
    if ask_yes_no("Would you like to copy the plugins?"):
        src_plugins = os.path.join(DIRECTORY_PATH, ".obsidian", "plugins")
        dest_plugins = os.path.join(OUTPUT_FOLDER, ".obsidian", "plugins")
        copy_directory(src_plugins, dest_plugins, folder_type="plugins")

    # Ask if the user wants to copy attachments
    if ask_yes_no("Would you like to copy over the attachments?"):
        app_json_path = os.path.join(DIRECTORY_PATH, ".obsidian", "app.json")
        if os.path.exists(app_json_path):
            attachment_folder = read_attachment_folder(app_json_path)
            src_attachments = os.path.join(DIRECTORY_PATH, attachment_folder)
            dest_attachments = os.path.join(OUTPUT_FOLDER, "Attachements")  # Corrected spelling to match the provided structure
            copy_directory(src_attachments, dest_attachments, folder_type="attachments")
        else:
            print(f"app.json not found at {app_json_path}")

    # Open Obsidian and switch to graph view after the output folder is ready
    open_obsidian_and_switch_to_graph(OUTPUT_FOLDER)
    
    display_summary(eligible_files, non_eligible_files, DIRECTORY_PATH, OUTPUT_FOLDER)

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
        DIRECTORY_PATH, 
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

if __name__ == "__main__":
    main()

