from flask import Flask
from flask import request
from flask import jsonify
from client.Client import allowed_file
from client.Client import convert_im_file, get_prediction_from_model

UPLOAD_FOLDER = '/temp'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/prediction", methods=['POST'])
def get_prediction():
    if request.method == 'POST':

        if 'image' not in request.files:
            print('No file part, key: image')
            return jsonify({"error": "No file part, key: image"})

        file = request.files['image']

        if file.filename == '':
            print('No selected file')
            return jsonify({"error": "No selected file"})

        if file and allowed_file(file.filename):
            img_decoded = convert_im_file(file)
            result = get_prediction_from_model(img_decoded)
            return jsonify({"result": result})

    return jsonify({"error": "500"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
