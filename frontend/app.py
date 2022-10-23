try:
    from flask import Flask, render_template
    from flask import request

    import uuid
    import os
    import json

    from datetime import datetime
    from flask import request

    from sentence_transformers import SentenceTransformer, util
    import boto3

    import numpy as np
    import elasticsearch
    from elasticsearch import Elasticsearch

    from dotenv import load_dotenv

    load_dotenv(".env")

except Exception as e:
    print("Error", e)

app = Flask(__name__)

model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')


class Tokenizer(object):
    def __init__(self):
        self.model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

    def get_token(self, documents):
        sentences = [documents]
        sentence_embeddings = self.model.encode(sentences)
        encod_np_array = np.array(sentence_embeddings)
        encod_list = encod_np_array.tolist()
        return encod_list[0]


class Search:
    def __init__(self, vector):
        self.vector = vector

    def get(self):
        query = {
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": "cosineSimilarity(params.queryVector, doc['vector'])+1.0",
                        "params": {
                            "queryVector": self.vector
                        }
                    }
                }
            }
        }

        es = Elasticsearch(hosts=[os.getenv("ELK_ENDPOINT")],
                           http_auth=(os.getenv("ELK_USERNAME"),
                                      os.getenv("ELK_PASSWORD")
                                      )
                           )

        res = es.search(index=os.getenv("ELK_INDEX"),
                        size=20,
                        body=query,
                        request_timeout=55)

        return res


@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")


@app.route("/ingest", methods=["GET", "POST"])
def ingest():
    return render_template("ingest.html")


@app.route("/get_results", methods=["GET", "POST"])
def get_results_data():
    try:
        request_data = dict(request.form)
        param = json.loads(request_data.get("data"))
        user_inputs = param.get("user_inputs")

        token_instance = Tokenizer()
        vector = token_instance.get_token(user_inputs)

        search_helper = Search(vector=vector)
        data = search_helper.get()

        data_cards = [x['_source'] for x in data['hits']['hits']]
        total_hits = 0

        try:
            total_hits = data['hits']['total']['value']
        except Exception as e:
            pass

        return {"data": data_cards, "total_hits": total_hits}

    except Exception as e:
        print("Error", e)
        return "error"


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True)
