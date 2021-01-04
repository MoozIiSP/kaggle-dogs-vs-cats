from flask import *
from .inference import Inferencer

from .config import *
from src.serving import inference


mod = Blueprint('api', __name__)


def check_post(args):
    for field in ['format', 'encoding']:
        if field not in args:
            return False, field
    return True, None


def process_data():
    pass


@mod.route('/v1/inference', methods = ['POST'])
def api_inference():
    if request.method == 'POST':
        meta = json.loads(request.form['meta'])
        ok, msg = check_post(meta.keys())
        if not ok:
            return jsonify({'status': f'{msg} is missed.'})

        files = request.files.getlist('data')
        if len(files) <= 0:
            return jsonify({'status': "image not found."})

        inferencer = Inferencer()
        results = {}
        for i, f in enumerate(files):
            results[f.filename] = inferencer.eval(f.read())
        print(results)
        return jsonify({'status': 'OK', 'scores': ''})
    else:
        return jsonify({'status': 'Wrong'})
