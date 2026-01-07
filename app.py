import pickle
import numpy as np
import json
from flask import Flask, request, render_template

# Load the trained components
model = pickle.load(open('components/random_forest_model.pkl', 'rb'))
encoder = pickle.load(open('components/oh_encoder.pkl', 'rb'))
pca = pickle.load(open('components/pca.pkl', 'rb'))
scaler = pickle.load(open('components/scaler.pkl', 'rb'))

label_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}

app = Flask(__name__)

def load_album_types():
    json_file = 'data/album_type_options.json'
    with open(json_file, "r", encoding="utf-8") as file:
        return json.load(file)
    
@app.route('/')
def home():
    album_types = load_album_types()
    return render_template('index.html', album_types=album_types)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None

    if request.method == 'POST':
        # numerical inputs
        album_release_year = int(request.form['year'])
        track_number = int(request.form['track-number'])
        artist_popularity = int(request.form['artist-popularity'])
        artist_followers = int(request.form['artist-followers'])
        album_total_tracks = int(request.form['album-total-tracks'])
        track_duration_min = float(request.form['track-duration-min'])

        # categorical inputs
        artist_genres = request.form['artist-genre']
        album_type = request.form['album-type']
        explicit = np.array([[int(request.form['explicit'])]])

        numerical_features = np.array([[
            track_number,
            artist_popularity,
            artist_followers,
            album_total_tracks,
            track_duration_min,
            album_release_year
        ]])

        categorical_features = [[artist_genres, album_type]]

        # scaling & encoding
        scaled_num_features = scaler.transform(numerical_features)
        encoded_features = encoder.transform(categorical_features)

        # concatenate
        data = np.hstack([scaled_num_features, explicit, encoded_features])

        # dimensionality reduction
        data_pca = pca.transform(data)

        # Make prediction
        predicted_class = model.predict(data_pca)[0]
        prediction = label_mapping[predicted_class]

    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
