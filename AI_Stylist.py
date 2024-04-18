import pandas as pd # Importing pandas for data processing, CSV file I/O
import json
import pandas as pd
from PIL import Image
import os
import google.generativeai as genai

cwd = os.getcwd()

def load_model():
    apiKey = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
    genai.configure(api_key = apiKey)

    model = genai.GenerativeModel('gemini-pro-vision')
    return model

def take_data():    
    images_df = pd.read_csv(cwd + "/static/images.csv")
    styles_df = pd.read_csv(cwd + "/static/styles.csv", on_bad_lines='skip')

    images_df['id'] = images_df['filename'].apply(lambda x: x.replace(".jpg", "")).astype(int)

    data = styles_df.merge(images_df, on='id', how='left').reset_index(drop=True)

    data['filename'] = data['filename'].apply(lambda x: os.path.join(cwd + "/static/images/", x))
    new_data = data[0:44424] # number of items in the database

    return new_data


def load_image_from_dataset(product_id, dataframe):
    try:
        row = dataframe[dataframe['id'] == product_id].iloc[0]
        file_path = row['filename']
        return Image.open(file_path)
    except IndexError:
        print(f"Product ID {product_id} not found in the dataset.")
        return None
    except FileNotFoundError:
        print(f"File not found for product ID {product_id}.")
        return None


def get_combination_feedback(user_selected_product_ids, data, model):
    selected_images = [load_image_from_dataset(pid, data) for pid in user_selected_product_ids]
    contents = ["Here are images of the selected products:"]

    for i, img in enumerate(selected_images):
        product_entry = f"Product {i+1}:"
        image_entry = img
        contents.extend([product_entry, image_entry])

    question =  "Consider the style, color, material, and overall appearance to determine if these products can be combined for an appealing look. If your decision is yes and the products are intended for different ages or genders, kindly highlight the age or gender differences for the customer and provide a conclusion to raise awareness before the customer makes a purchase in your reason. Return in JSON with decision and reason:"
    contents.append(question)

    # Call to the generative AI model
    responses = model.generate_content(contents, stream=True)
    all_responses = [response.text for response in responses]
    print(all_responses)
    decision, reason = parse_response(all_responses)
    return decision, reason


def parse_response(response_parts):
    try:
        concatenated_response = ''.join(response_parts)
        cleaned_response = concatenated_response.replace('```json', '').replace('```', '').strip()
        response_json = json.loads(cleaned_response)
        decision = response_json.get("decision", "Unknown")
        reason = response_json.get("reason", "No reason provided")
        return decision, reason
    except json.JSONDecodeError:
        return "Unknown", "Response parsing error"
    