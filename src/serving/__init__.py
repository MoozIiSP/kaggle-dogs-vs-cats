from flask import Flask
from flask.json import jsonify
from .config import *


app = Flask("hydra_serving")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.errorhandler(404)
def not_found(error):
    return jsonify({"status": "404"}), 404

from . import routers
app.register_blueprint(routers.mod)

# Initializing inferencer
from .inference import Inferencer
gInferencer = Inferencer()
