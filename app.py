import os
import numpy as np
from numpy.linalg import norm
import pickle
import tensorflow

from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

from flask import Flask, render_template, request, session, redirect, url_for, jsonify

import recommended_system
import AI_Stylist

import pandas as pd
from PIL import Image

from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__, static_url_path='/static')
app.secret_key = 'trunghau'  # Set a secret key for session management
cwd = os.getcwd() # current directory

styles_df = pd.read_csv(cwd + "/static/styles.csv", on_bad_lines='skip')

# Load data features
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

# Add layer
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])


def clear_cart():
    session.pop('cart', None)

@app.before_request
def before_request():
    if 'cart' not in session:
        clear_cart()

# Front site
@app.route('/')
def home():

    # More product
    more_product_df = styles_df
    more_product_df = more_product_df.sample(frac=1)
    top_10_more_product = more_product_df.head(7)
    more_products = top_10_more_product['id'].tolist()

    # Men pick
    men_df = styles_df[styles_df['gender'] == 'Men']
    men_df = men_df.sample(frac=1)
    top_10_men = men_df.head(7)
    men_ids = top_10_men['id'].tolist()

    # Women pick
    women_df = styles_df[styles_df['gender'] == 'Women']
    women_df = women_df.sample(frac=1)
    top_10_women = women_df.head(7)
    women_ids = top_10_women['id'].tolist()

    # Kids pick
    boys_df = styles_df[styles_df['gender'] == 'Boys']
    boys_df = boys_df.sample(frac=1)
    top_5_boys = boys_df.head(3)
    top_5_boys_ids = top_5_boys['id'].tolist()

    girls_df = styles_df[styles_df['gender'] == 'Girls']
    girls_df = girls_df.sample(frac=1)
    top_5_girls = girls_df.head(4)
    top_5_girls_ids = top_5_girls['id'].tolist()

    kids_ids = top_5_boys_ids + top_5_girls_ids

    # Shoes pick
    shoes_styles_df = styles_df[(styles_df['masterCategory'] == 'Footwear') & 
              (styles_df['subCategory'] == 'Shoes')]
    shoes_styles_df = shoes_styles_df.sample(frac=1)
    top_10_shoes_id = shoes_styles_df.head(7)
    shoes_ids = top_10_shoes_id['id'].tolist()

    # Jewellery pick
    jewellery_styles_df = styles_df[styles_df['subCategory'] == 'Jewellery']
    jewellery_styles_df = jewellery_styles_df.sample(frac=1)
    top_10_jewellery_id = jewellery_styles_df.head(7)
    jewellery_ids = top_10_jewellery_id['id'].tolist()

    # Accessories picks
    accessories_styles_df = styles_df[(styles_df['masterCategory'] == 'Accessories') &
                                      (styles_df['subCategory'] != 'Jewellery')]
    accessories_styles_df = accessories_styles_df.sample(frac=1)
    top_10_accessories = accessories_styles_df.head(7)
    accessories_ids = top_10_accessories['id'].tolist()

    # Zip image and image's info
    def generate_zipped_lists(ids_list, category):
        path_list = [url_for('static', filename=os.path.join('images', str(name) + '.jpg')) for name in ids_list]
        display_name_list = [styles_df.loc[styles_df['id'] == int(name), 'productDisplayName'].values[0] for name in ids_list]
        zipped_list = zip(path_list, display_name_list)
        return zipped_list
    
    
    more_products_zipped = generate_zipped_lists(more_products, 'more_products')
    men_zipped = generate_zipped_lists(men_ids, 'men')
    women_zipped = generate_zipped_lists(women_ids, 'women')
    kids_zipped = generate_zipped_lists(kids_ids, 'kids')
    shoes_zipped = generate_zipped_lists(shoes_ids, 'shoes')
    jewellery_zipped = generate_zipped_lists(jewellery_ids, 'jewellery')
    accessories_zipped = generate_zipped_lists(accessories_ids, 'accessories')
    
    return render_template('index.html', more_products=more_products_zipped, men_products=men_zipped, women_products=women_zipped, 
                           kid_products=kids_zipped, shoes_products=shoes_zipped, jewellery_products=jewellery_zipped, accessories_products=accessories_zipped)


# Recommendation site
@app.route('/result', methods=['POST'])
def upload_file():
    if 'file' in request.files:
        file = request.files['file']
        filename = file.filename
        
        file_path = os.path.join('static', 'images', filename)

        # For items belong to database and visable in the site
        if os.path.exists(file_path):

            file.save(file_path)
            # Get the URL of the uploaded image
            image_url = url_for('static', filename=os.path.join('images', filename))
            print("URL of the uploaded image:", image_url)

            id = filename.split(".")[0]
            product_display_names = styles_df.loc[styles_df['id'] == int(id), ['productDisplayName', 'gender', 'baseColour', 'masterCategory', 'subCategory']].values[0] # 0: Name, 1: Gender, 2: Color, 3: Category, 4: Sub Category
            
            # Perform feature extraction and recommendation for the chosen product
            features_chosen = recommended_system.feature_extraction(file_path, model)
            indices = recommended_system.recommend(features_chosen, feature_list, 9)
            recommended_products = [filenames[idx] for idx in indices[0][1:]]
            
            # Get the paths of recommended product images & zip the image and image's info
            image_paths = [url_for('static', filename=name) for name in recommended_products]
            display_recommended_name_list = [styles_df.loc[styles_df['id'] == int(name.split('/')[-1].split('.')[0]), 'productDisplayName'].values[0] for name in image_paths]
            recommended_zipped = zip(image_paths, display_recommended_name_list)
            print(image_paths)
            
            # Pass the URL of the chosen product and URLs of recommended products to the template
            return render_template('result.html', chosen_product=[image_url], product_display_names=product_display_names, recommended_zipped=recommended_zipped)
        
        # For items not belong to database
        else:
            
            # Read the uploaded image
            img = Image.open(file.stream)

            def feature_extraction(img, model):
                img_resized = img.resize((224, 224))
                img_array = image.img_to_array(img_resized)
                expanded_img_array = np.expand_dims(img_array, axis=0)
                preprocessed_img = preprocess_input(expanded_img_array)
                result = model.predict(preprocessed_img).flatten()
                normalized_result = result / norm(result)
                return normalized_result
            
            # Perform feature extraction and recommendation for the uploaded image
            features_chosen = feature_extraction(img, model)
            indices = recommended_system.recommend(features_chosen, feature_list, 8)

            recommended_features = feature_list[indices[0]]
            similarity_values = cosine_similarity([features_chosen], recommended_features)
            print(similarity_values)

            recommended_products = [filenames[idx] for idx in indices[0][0:]]
            product_display_names = ['Unknown product', '-', '-', '-', '-']
            
            # Get the paths of recommended product images
            image_paths = [url_for('static', filename=name) for name in recommended_products]
            display_recommended_name_list = [styles_df.loc[styles_df['id'] == int(name.split('/')[-1].split('.')[0]), 'productDisplayName'].values[0] for name in image_paths]
            recommended_zipped = zip(image_paths, display_recommended_name_list)
            print(image_paths)

            # Render the result template with recommended product images
            return render_template('result.html', chosen_product=[img], product_display_names=product_display_names, recommended_zipped=recommended_zipped)
    

    # For items belong to database and unvisable in the site
    elif 'filename' in request.form:
        # If the filename is provided in the form, construct the file path
        filename = request.form['filename']
        file_path = os.path.join('static', 'images', filename)

        id = filename.split(".")[0]
        product_display_names = styles_df.loc[styles_df['id'] == int(id), ['productDisplayName', 'gender', 'baseColour', 'masterCategory', 'subCategory']].values[0] # 0: Name, 1: Gender, 2: Color, 3: Category, 4: Sub Category
        
        # Perform feature extraction and recommendation for the chosen product
        features_chosen = recommended_system.feature_extraction(file_path, model)
        indices = recommended_system.recommend(features_chosen, feature_list, 9)
        recommended_products = [filenames[idx] for idx in indices[0][1:]]

        # Get the paths of recommended product images
        image_paths = [url_for('static', filename=name) for name in recommended_products]
        display_recommended_name_list = [styles_df.loc[styles_df['id'] == int(name.split('/')[-1].split('.')[0]), 'productDisplayName'].values[0] for name in image_paths]
        recommended_zipped = zip(image_paths, display_recommended_name_list)
        print(image_paths)

        # Pass the URLs of recommended products to the template
        return render_template('result.html', chosen_product=[file_path], product_display_names=product_display_names, recommended_zipped=recommended_zipped)

    else:
        return redirect(url_for('home'))
    

# Add to cart feature
@app.route('/add_to_cart', methods=['POST'])
def add_to_cart():
    data = request.json
    filename = data.get('filename')
    product_name = filename.split(".")[0]
    if product_name:
        # Add the filename to the session cart list
        cart = session.get('cart', [])
        if product_name not in cart:
            if product_name != "PIL":
                cart.append(product_name)
            session['cart'] = cart
            print("Number of items in cart:", len(cart))
            print("Name of items in cart:", cart)
        
        display_name = styles_df.loc[styles_df['id'] == int(product_name), 'productDisplayName'].values[0]

        return jsonify({"product_name": display_name})
    else:
        return "Error: No filename provided"

    
# Count items on cart
@app.route('/cart_count')
def cart_count():
    cart = session.get('cart', [])
    print("Number of items in cart:", len(cart))
    print("Name of items in cart:", cart)
    return str(len(cart))


# Cart site
@app.route('/cart')
def cart():
    cart = session.get('cart', [])
    cart_with_extension = ['images/' + name + '.jpg' for name in cart]

    # Generate image paths with the updated filenames
    image_paths = [url_for('static', filename=name) for name in cart_with_extension]
    display_cart_name_list = [styles_df.loc[styles_df['id'] == int(name.split('/')[-1].split('.')[0]), 'productDisplayName'].values[0] for name in image_paths]
    recommended_zipped = zip(image_paths, display_cart_name_list)

    return render_template('cart.html', cart=cart, recommended_zipped=recommended_zipped, cart_items=cart_with_extension)


# Remove from cart feature
@app.route('/remove_from_cart', methods=['POST'])
def remove_from_cart():
    data = request.json
    path = data.get('path')
    path = os.path.basename(path)
    path = os.path.splitext(path)[0]

    # Remove the image path from the cart in the session
    cart = session.get('cart', [])
    if path in cart:
        cart.remove(path)
        session['cart'] = cart
    print(len(cart))

    return jsonify({
        "cart_length": str(len(cart)),
        "cart": cart
    })


# AI Stylist
@app.route('/ask_ai_stylist', methods=['POST'])
def ask_ai_stylist():
    model = AI_Stylist.load_model()
    new_data = AI_Stylist.take_data()
    
    # Take the id of the product from cart
    cart = session.get('cart', [])
    cart = [int(product_id) for product_id in cart]

    # Call AI Stylist function to get feedback
    decision, reason = AI_Stylist.get_combination_feedback(cart, new_data, model)
    print(f"AI Stylist's Decision: {decision}")
    print("Reason:", reason)

    return jsonify(decision, reason)


# Place your order feature
@app.route('/place_order', methods=['POST'])
def place_order():
    session.pop('cart', None)

    # Display a confirmation message
    return "Thank you for shopping with us!!"


if __name__ == '__main__':
    app.run(port=5000)