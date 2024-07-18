import argparse
import base64
import io
import json
import matplotlib.pyplot as plt
from gradio_client import Client
from PIL import Image


def check_text_toxicity(text, personalize_safer_value=0.005):
    client = Client("https://duchaba-friendly-text-moderation.hf.space/--replicas/36u4a/")
    try:
        result = client.predict(
            text,  # str in 'Enter Text:' Textbox component
            personalize_safer_value,  # float (numeric value between 0.005 and 0.1) in 'Personalize Safer Value: (larger value is less safe)' Slider component
            api_name="/censor_me"
        )
        return result
    except Exception as e:
        print(f"Error occurred while calling the API: {e}")
        return None


def save_plot(plot_data, output_file="toxicity_plot.png"):
    try:
        # Extract the base64 encoded image data
        image_data = plot_data['plot'].split(',')[1]
        
        # Decode the base64 data
        image_bytes = base64.b64decode(image_data)
        
        # Open the image using PIL
        image = Image.open(io.BytesIO(image_bytes))
        
        # Save the image
        image.save(output_file)
        print(f"Plot saved as {output_file}")
    except Exception as e:
        print(f"Error occurred while saving the plot: {e}")


def interpret_result(result):
    # This is a placeholder function. In a real-world scenario, you'd need to 
    # know the exact structure of the result to provide a meaningful interpretation.
    print("\nToxicity Analysis Interpretation:")
    print("--------------------------------")
    print("The toxicity analysis is represented in the saved plot.")
    print("Red areas indicate higher toxicity, while green areas indicate lower toxicity.")
    print("The exact meanings of the plot sections would depend on the API's specific categorizations.")


def main():
    parser = argparse.ArgumentParser(description="Check text toxicity using Friendly Text Moderation API")
    parser.add_argument("text", type=str, help="The text to check for toxicity")
    parser.add_argument("--safe-value", type=float, default=0.005, help="Personalize safer value (default: 0.005)")
    parser.add_argument("--output", type=str, default="toxicity_plot.png", help="Output file name for the plot")
    
    args = parser.parse_args()
    
    result = check_text_toxicity(args.text, args.safe_value)
    
    if result:
        print("Toxicity analysis completed successfully.")
        save_plot(result, args.output)
        interpret_result(result)
    else:
        print("Failed to perform toxicity analysis.")


if __name__ == "__main__":
    main()
