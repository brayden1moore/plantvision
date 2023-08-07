import plantvision
import requests
from io import BytesIO
import pickle as pkl
from flask import Flask, render_template, request, session, jsonify, url_for
from PIL import Image
import os
import time
import random
from pathlib import Path
THIS_FOLDER = Path(__file__).parent.resolve()

app = Flask(__name__)
app.secret_key = 'pi-33pp-co-sk-33'
app.template_folder = os.path.abspath(f'{THIS_FOLDER}/web/templates')
app.static_folder = os.path.abspath(f'{THIS_FOLDER}/web/static')
print(app.static_folder)

flowerLayers = None
leafLayers = None
fruitLayers = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/guess', methods=['POST'])
def guess():
    global flowerLayers, leafLayers, fruitLayers

    if request.method == 'POST':
        print('Thinking...')
        
        img = request.files.get('uploaded-image')
        feature = request.form.get('feature')
        
        tensor = plantvision.processImage(img, feature)
        predictions = plantvision.see(tensor, feature, 6)

        with open(f'{THIS_FOLDER}/resources/speciesNameToKey.pkl','rb') as f:
            speciesNameToKey = pkl.load(f)
        with open(f'{THIS_FOLDER}/resources/speciesNameToVernacular.pkl','rb') as f:
            speciesNameToVernacular = pkl.load(f)
        with open(f'{THIS_FOLDER}/resources/{feature}speciesIndexDict.pkl','rb') as f:
            speciesNameToIndex = pkl.load(f)

        urls = []
        predicted_image_urls = []
        for p in predictions:
            key = speciesNameToKey[p]
            img = speciesNameToIndex[p]
            query = ''
            for i in p.split(' '):
                query += i 
                query += '+'
            urls.append(f'https://www.google.com/search?q={query[:-1]}')
            predicted_image_urls.append(f"https://storage.googleapis.com/bmllc-images-bucket/images/img{img}.jpeg")

        names = []
        for p in predictions:
            try:
                names.append(speciesNameToVernacular[p])
            except:
                names.append(p)

        response = {
            'names': names,
            'species': predictions,
            'predictions': urls,
            'images': predicted_image_urls
        }

        return jsonify(response)

if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 8080)),host='0.0.0.0',debug=True)
