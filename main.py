from gcloud import storage
from oauth2client.service_account import ServiceAccountCredentials
import os

credentials_dict = {
    'type': 'service_account',
    'client_id': os.environ['CLIENT_ID'],
    'client_email': os.environ['CLIENT_EMAIL'],
    'private_key_id': os.environ['PRIVATE_KEY_ID'],
    'private_key': os.environ['PRIVATE_KEY'].replace('\\n', '\n'),
}

credentials = ServiceAccountCredentials.from_json_keyfile_dict(
    credentials_dict
)
client = storage.Client(credentials=credentials, project='bmllc-plant')
bucket = client.get_bucket('bmllc-plant-image-bucket')

import plantvision
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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/guess', methods=['POST'])
def guess():

    session['sessionId'] = random.random()*100
    if request.method == 'POST':
        print('Thinking...')
        img = request.files.get('uploaded-image')
        feature = request.form.get('feature')

        tensor = plantvision.processImage(img, feature)
        predictions = plantvision.see(tensor, feature, 6)
        #confidences = [f'{str(round(i*100,4))}%' for i in confidences]

        with open(f'{THIS_FOLDER}/resources/speciesNameToKey.pkl','rb') as f:
            speciesNameToKey = pkl.load(f)
        with open(f'{THIS_FOLDER}/resources/speciesNameToVernacular.pkl','rb') as f:
            speciesNameToVernacular = pkl.load(f)
        with open(f'{THIS_FOLDER}/resources/{feature}speciesIndexDict.pkl','rb') as f:
            speciesNameToIndex = pkl.load(f)

        urls = []
        predictedImages = []
        predicted_image_urls = []
        for p in predictions:
            key = speciesNameToKey[p]
            img = speciesNameToIndex[p]
            query = ''
            for i in p.split(' '):
                query += i 
                query += '+'
            urls.append(f'https://www.google.com/search?q={query[:-1]}')
            #predictedImages.append(f'{THIS_FOLDER}/images/img{img}.jpeg')
            predicted_image_urls.append(f"https://storage.googleapis.com/bmllc-images-bucket/images/img{img}.jpeg")
        
        #predicted_image_urls = []
        #for i,image in enumerate(predictedImages):
        #    blob = bucket.blob(f"{session['sessionId']}_{i}.jpeg")
        #    blob.upload_from_filename(image)
        #    predicted_image_urls.append(f"https://storage.googleapis.com/bmllc-plant-image-bucket/{session['sessionId']}_{i}.jpeg")

        #urls = []
        #predictedImages = []
        #for p in predictions:
        #    key = speciesNameToKey[p]
        #    img = speciesNameToIndex[p]
        #    query = ''
        #    for i in p.split(' '):
        #        query += i
        #        query += '+'
        #    urls.append(f'https://www.google.com/search?q={query[:-1]}')
        #    #urls.append(f'https://www.gbif.org/species/{key}')
        #    predictedImages.append(Image.open(f'{THIS_FOLDER}/images-highres/img{img}.jpeg'))

        #for i,image in enumerate(predictedImages):
        #    image.save(f'{THIS_FOLDER}/web/static/predicted-images/{session['sessionId']}_{i}.jpeg', "JPEG")

        #predicted_image_urls = [url_for(f'static', filename=f'predicted-images/{session['sessionId']}_{i}.jpeg') for i in range(len(predictedImages))]

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
