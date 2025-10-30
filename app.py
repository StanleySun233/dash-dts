import json

from flask import Flask, request, jsonify, render_template

from dialogue_dataset import DialogueDataset
from metrics import (
    evaluate_segmentation
)
from model.DSAgent import DSAgent
from model.DTSAgent import DTSAgent
from model.HSAgent import HSAgent
from model.LLMReassessmentAgent import LLMReassessmentAgent
from model.PNAgent import PNAgent
from utils import load_config, resolve_dataset_path
import os
from impl import (
    list_dialogues,
    get_dialogue_info,
    run_handshake,
    run_posneg,
    run_dts,
    run_similarity,
    compute_metrics,
    run_reassess,
    run_segment,
    run_reassess_by_id,
)

# Create Flask app
app = Flask(__name__)

# Load configuration from config file
config = load_config("config.yaml")
api_key = config["api_key"]["openrouter"]
base_url = config["base_url"]["openrouter"]
model = config["model"]["openrouter"][0]
window_size = config.get("window_size", 3)

# Initialize dataset for agents (supports env DATASET or default 'vfh')
_dataset_name = os.environ.get('DATASET', 'vfh')
_dataset_path = resolve_dataset_path(_dataset_name)
dataset = DialogueDataset(_dataset_path)

# Initialize Agents
hs_agent = HSAgent(dataset, api_key, base_url, model, window_size=3)
pn_agent = PNAgent(dataset, api_key, base_url, model, window_size=3)
dts_agent = DTSAgent(dataset, api_key, base_url, model, window_size=3)
ds_agent = DSAgent(dataset)
reassessment_agent = LLMReassessmentAgent(api_key, base_url, model)



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/dialogues', methods=['GET'])
def get_dialogues():
    try:
        result = list_dialogues(dataset)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/dialogue/<dial_id>', methods=['GET'])
def get_dialogue(dial_id):
    try:
        result, status = get_dialogue_info(dataset, dial_id)
        if status != 200:
            return jsonify(result), status
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/handshake', methods=['POST'])
def handshake():
    try:
        data = request.get_json()
        result, status = run_handshake(hs_agent, data)
        return jsonify(result), status

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/posneg', methods=['POST'])
def posneg():
    try:
        data = request.get_json()
        result, status = run_posneg(pn_agent, data)
        return jsonify(result), status

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/dts', methods=['POST'])
def dts():
    try:
        data = request.get_json()
        result, status = run_dts(dts_agent, data)
        return jsonify(result), status

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/similarity', methods=['POST'])
def similarity():
    try:
        data = request.get_json()
        result, status = run_similarity(ds_agent, data["dialogue"])
        return jsonify(result), status

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/metrics', methods=['POST'])
def metrics():
    try:
        data = request.get_json()
        result, status = compute_metrics(data["reference"], data["hypothesis"])
        return jsonify(result), status

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/reassess', methods=['POST'])
def reassess():
    try:
        data = request.get_json()
        content = data['content']
        prediction = data.get('prediction')
        result, status = run_reassess(reassessment_agent, content, prediction)
        return jsonify(result), status

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/segment', methods=['POST'])
def segment_dialogue():
    try:
        data = request.get_json()
        dial_id = data.get('dial_id')
        handshake_results = data.get('handshake_results', None)
        few_shot_examples = data.get('few_shot_examples', None)
        similarity_examples = data.get('similarity_examples', None)
        result, status = run_segment(dataset, dts_agent, dial_id, handshake_results, few_shot_examples, similarity_examples)
        return jsonify(result), status

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/reassess_dialogue', methods=['POST'])
def reassess_dialogue():
    try:
        data = request.get_json()
        dial_id = data.get('dial_id')
        prediction = data.get('prediction')
        result, status = run_reassess_by_id(dataset, reassessment_agent, dial_id, prediction)
        return jsonify(result), status

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=os.environ.get('DATASET', 'vfh'),
                        help='vfh | dialseg_711 | doc2dial or path to json')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    args = parser.parse_args()

    # Re-init dataset/agents if CLI overrides
    ds_path = resolve_dataset_path(args.dataset)
    if ds_path != _dataset_path:
        dataset = DialogueDataset(ds_path)
        hs_agent = HSAgent(dataset, api_key, base_url, model, window_size=3)
        pn_agent = PNAgent(dataset, api_key, base_url, model, window_size=3)
        dts_agent = DTSAgent(dataset, api_key, base_url, model, window_size=3)
        ds_agent = DSAgent(dataset)

    app.run(debug=True, host=args.host, port=args.port)
