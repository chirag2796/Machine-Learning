from flask import Flask, send_from_directory, send_file
from flask_cors import CORS, cross_origin
app = Flask(__name__, static_url_path='')
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
import json

data = None
with open("browser\\models\\model-big\\model.json", "r") as read_file:
    data = json.load(read_file)


@app.route('/')
@cross_origin()
def get_model():
    return data

@app.route('/group1-shard1of1.bin')
def fun():
    try:
        return send_file('browser/models/model-big/group1-shard1of1.bin',
                         attachment_filename='group1-shard1of1.bin')
    except Exception:
        return 'ERROR'

# @app.route('/class_names')
# def fun_class_names():
#     try:
#         return send_file('browser/models/model-big/class_names.txt',
#                          attachment_filename='class_names.txt')
#     except Exception:
#         return 'ERROR'

# @app.route('/browser/model2/<path:path>')
# def send_bin(path):
#     return send_from_directory('browser\\model2', path)

if __name__ == '__main__':
    app.run(debug=True)