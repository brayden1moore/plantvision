import plantvision
import pickle as pkl
from flask import Flask, render_template, request, session, jsonify, url_for
from PIL import Image
import os
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
        predictions, confidences = plantvision.see(tensor, feature, 9)
        confidences = [f'{str(round(i*100,4))}%' for i in confidences]

        with open(f'{THIS_FOLDER}/resources/speciesNameToKey.pkl','rb') as f:
            speciesNameToKey = pkl.load(f)
        with open(f'{THIS_FOLDER}/resources/speciesNameToVernacular.pkl','rb') as f:
            speciesNameToVernacular = pkl.load(f)
        with open(f'{THIS_FOLDER}/resources/{feature}speciesIndexDict.pkl','rb') as f:
            speciesNameToIndex = pkl.load(f)

        urls = []
        predictedImages = []
        for p in predictions:
            key = speciesNameToKey[p]
            img = speciesNameToIndex[p]
            query = ''
            for i in p.split(' '):
                query += i 
                query += '+'
            urls.append(f'https://www.google.com/search?q={query[:-1]}')
            #urls.append(f'https://www.gbif.org/species/{key}')
            predictedImages.append(Image.open(f'{THIS_FOLDER}/images/img{img}.jpeg'))

        for i,image in enumerate(predictedImages):
            image.save(f'{THIS_FOLDER}/web/static/predicted-images/img{i}.jpeg', "JPEG")

        predicted_image_urls = [url_for(f'static', filename=f'predicted-images/img{i}.jpeg') for i in range(len(predictedImages))]

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
            'images': predicted_image_urls,
            'confidences': confidences
        }

        return jsonify(response)

if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 8080)),host='0.0.0.0',debug=True)
