import base64
import json
import re
from PIL import Image
import io
import pytesseract

# Function to read JSON file
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

response_data = read_json_file('response.json')


def analyze_toxicity_response(response_data):
    insights = {}

    # Extract image type
    insights['image_type'] = response_data['type']

    # Extract base64 image data
    base64_image = response_data['plot'].split(',')[1]

    # Decode base64 to image
    image_data = base64.b64decode(base64_image)
    image = Image.open(io.BytesIO(image_data))

    # Use OCR to extract text from image
    text = pytesseract.image_to_string(image)
    print("Extracted text:", text)  # For debugging

    # Extract insights from the OCR text
    classification_match = re.search(r'(Unsafe|Safe):?\s*(\w+):?\s*([\d.]+)%\s*Confidence', text)
    if classification_match:
        insights['overall_status'] = classification_match.group(1)
        insights['classification'] = classification_match.group(2)
        insights['confidence'] = float(classification_match.group(3))
    else:
        insights['overall_status'] = 'Unknown'
        insights['classification'] = 'Unknown'
        insights['confidence'] = 0.0

    percentage_match = re.findall(r'([\d.]+)%', text)
    if len(percentage_match) >= 2:
        insights['primary_category_percentage'] = float(percentage_match[0])
        insights['other_categories_percentage'] = float(percentage_match[1])
    else:
        insights['primary_category_percentage'] = 0.0
        insights['other_categories_percentage'] = 0.0

    category_match = re.search(r'(\w+)\s+[\d.]+%', text)
    if category_match:
        insights['primary_category'] = category_match.group(1)
    else:
        insights['primary_category'] = 'Unknown'

    other_categories_match = re.search(r'Other\s+(\d+)\s+categories', text)
    if other_categories_match:
        insights['other_categories_count'] = int(other_categories_match.group(1))
    else:
        insights['other_categories_count'] = 0

    return insights

# Example usage:
# response_data = {
#     'type': 'matplotlib',
#     'plot': 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAoAAAAHgCAYAAAA10dzkAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8fJSN1AAAACXBIWXMAAA9hAAAPYQGoP6dpAABoqElEQVR4nO3dd3yM9wMH8M/lsiMyJAQhJELsEVTVpqg9SnWo2IqiLZ1+RUvtqj2qVbPVoqitRu1Zq2YkYsaILSHJ3X1/fzxyyUmQcHffu3s+79crL8/dPbn7XFwun/t+n6ERQggQERERkWo4yQ5ARERERNbFAkhERESkMiyARERERCrDAkhERESkMiyARERERCrDAkhERESkMiyARERERCrDAkhERESkMiyARERERCrDAkhERESkMiyARERERCrDAkhERESkMiyARERERCrDAkhERESkMiyARERERCrDAkhERESkMiyARERERCrDAkhERESkMiyARERERCrDAkhERESkMiyARERERCrDAkhERESkMiyARERERCrDAkhERESkMiyARERERCrDAkhERESkMiyARERERCrDAkhERESkMiyARERERCrDAkhERESkMiyARERERCrDAkhERESkMiyARERERCrDAkhERESkMiyARERERCrDAkhERESkMiyA...'  # Your full base64 string here
# }

insights = analyze_toxicity_response(response_data)
print(json.dumps(insights, indent=2))