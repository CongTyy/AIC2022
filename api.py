from flask import Flask, stream_with_context, Response, render_template
import os
import shutil
from main import Main
import argparse

app = Flask(__name__)

main = Main()

def page_generator():
    print("Preprocessing data")
    embs, dataset_dict = main.pre_processing()
    while True:
        output, query = main.inference(embs, dataset_dict)
        yield render_template("api.html", dic = output, len = len(output['paths']), name = query)

generator_obj = None

@app.route("/", methods=['POST', 'GET'])
def home():
    global generator_obj
    generator_obj = generator_obj or page_generator()
    return Response(stream_with_context(next(generator_obj)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--port', default= 5000)

    args = parser.parse_args()

    _port = args.port
    app.run(host='0.0.0.0', debug=False, port = _port)
