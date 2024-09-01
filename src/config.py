import os
import multiprocessing

MODEL = 'llama3'
TEMPERATURE = 0.0
MODEL_API_URL = 'http://localhost:11434'
MAX_FILE_SIZE = 20000
MIN_FILE_SIZE = 10
DIRECTORY_PATH = r'E:\Documents\Obsidian\My Vault'
OUTPUT_FOLDER = r'E:\Documents\Obsidian\Obsidian_Enhanced'
LOG_DIRECTORY = 'logs'
PROCESS_LOG_FILE = 'process_log.txt'
ERROR_LOG_FILE = 'process_errors_log.txt'
PROMPTS_CONFIG_FILE = 'prompts_config.json'
VERBOSE = False
MAX_WORKERS = min(8, multiprocessing.cpu_count())  # Adjust based on the number of CPU cores
CHECKPOINT = False
MAX_RETRIES = 3  # Maximum number of retries for API requests
RETRY_DELAY = 2  # Delay between retries in seconds

# Flags for Final Check
llm_communication = False
attachments_transferred = False
plugins_transferred = False

