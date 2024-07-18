# Toxicity Checker & Toxicity Language

This repository contains scripts for checking the toxicity of text across multiple languages using the Friendly Text Moderation API and OpenAI's GPT-4 for translation. The results are classified into three categories based on confidence levels.

## Classifications
1. **Low Confidence:** Confidence is less than 70, regardless of the overall status.
2. **Borderline:** Confidence is less than 90 and overall status is 'Unsafe'.
3. **Detected:** All other cases.

## Files
1. `toxicity_checker.py`
2. `toxicity_language.py`

### `toxicity_checker.py`
This script is responsible for checking the toxicity of a given text in multiple languages. It translates the text, checks its toxicity, interprets the results, and generates a detailed report.

#### Usage
```bash
python toxicity_checker.py "your text to check" --safe-value 0.005 --output toxicity_report.txt

## Overview

This project provides a tool to check the toxicity of text using the Friendly Text Moderation API. The tool analyzes the input text for toxic content and generates a visual representation of the analysis.

## Project Structure

```
text_toxicity_checker/
├── toxicity_checker.py
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/text_toxicity_checker.git
   cd text_toxicity_checker
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script to check the toxicity of text:

    ```bash
    python toxicity_checker.py "Your text here" --safe-value 0.005 --output toxicity_plot.png
    ```

### Arguments

- `text`: The text to check for toxicity.
- `--safe-value`: Personalize safer value (default: 0.005).
- `--output`: Output file name for the plot (default: `toxicity_plot.png`).

### Example

    ```bash
    python toxicity_checker.py "This is a sample text." --safe-value 0.01 --output sample_plot.png
    ```

## Components

- `toxicity_checker.py`: Main script to check text toxicity, save the resulting plot, and interpret the results.

### Functions

- `check_text_toxicity(text, personalize_safer_value=0.005)`: Calls the Friendly Text Moderation API to check the toxicity of the provided text.
- `save_plot(plot_data, output_file="toxicity_plot.png")`: Saves the toxicity analysis plot to a file.
- `interpret_result(result)`: Provides a basic interpretation of the toxicity analysis result.

## Configuration

The script uses the Friendly Text Moderation API. Ensure you have access to the API endpoint and adjust the API URL if needed in the `check_text_toxicity` function.

## Results

The project generates an output file (`toxicity_plot.png` by default) that visually represents the toxicity analysis of the input text.

## Ethical Considerations

This tool is designed for research and educational purposes. Use this tool responsibly and ethically. The creation and use of toxic content can be harmful.

## Contributors

Joe Tustin

## Acknowledgments

- Thanks to the creators of the Friendly Text Moderation API.
- Thanks to the developers of the Gradio Client and other dependencies used in this project.

Feel free to reach out with any questions or feedback!
