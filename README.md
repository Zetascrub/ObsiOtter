# ObsiOtter

<div align="center">
    <img src="./logo.png" alt="ObsiOtter Logo" width="35%"/>
</div>

ObsiOtter is a powerful tool designed to refine and process Markdown files in your Obsidian vault. Leveraging the Ollama LLM, it transforms raw notes into well-structured, enhanced documents, ready for use in Obsidian.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features
- **Automated Markdown Processing:** Easily process and refine text files.
- **Integration with Obsidian:** Copy and manage plugins and attachments for seamless Obsidian integration.
- **Parallel Processing:** Efficiently handle large batches of files with parallel processing.
- **LLM-Powered Enhancements:** Use AI to enhance the content of your notes.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/obsiotter.git
   cd obsiotter
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Before running the script, configure the necessary parameters to match your environment:

```bash
MODEL_API_URL: The URL for the Ollama LLM.
DIRECTORY_PATH: The directory containing the files to be processed.
OUTPUT_FOLDER: The directory where processed files will be saved.
```

## Usage

1. Run the script:
   ```bash
   python ObsiOtter.py
   ```

2. Follow the prompts to configure your processing options.

## Contributing

Feel free to submit issues or pull requests if you have suggestions or improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This project was inspired by the work done in [Obsidian-Text-Transformer-LocalAI](https://github.com/Amine-LG/Obsidian-Text-Transformer-LocalAI) by Amine-LG. Their approach to integrating AI with Obsidian served as a valuable foundation for developing ObsiOtter. We extend our gratitude for their contributions to the community.
