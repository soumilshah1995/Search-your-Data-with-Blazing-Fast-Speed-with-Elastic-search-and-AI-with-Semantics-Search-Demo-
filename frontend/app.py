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

global model
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')


class Tokenizer(object):

    @staticmethod
    def get_token(documents):
        sentences = [documents]
        sentence_embeddings = model.encode(sentences)
        encod_np_array = np.array(sentence_embeddings)
        encod_list = encod_np_array.tolist()
        return encod_list[0]


def create_scroll(raw_response):
    try:
        data = raw_response.get("hits", None).get("hits", None)
        if not data: return None
        data = data[-1]
        score = data.get("_score", 1)
        scroll_id_ = data.get("_id", None)
        unique_scroll_id = "{},{}".format(score, scroll_id_)
        return unique_scroll_id

    except Exception as e:
        return ""


class Search(object):
    def __init__(self, user_query):
        self.user_query = user_query
        self.vector = Tokenizer.get_token(documents=self.user_query)

        self.es = Elasticsearch(hosts=[os.getenv("ELK_ENDPOINT")],
                                http_auth=(os.getenv("ELK_USERNAME"),
                                           os.getenv("ELK_PASSWORD")
                                           )
                                )

    def get_query(self):
        return {
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

    def search(self, size=20, scroll_id=None):
        if scroll_id is None:
            res = self.es.search(index=os.getenv("ELK_INDEX"),
                                 size=size,
                                 body=self.get_query(),
                                 request_timeout=55)

            scroll_id = create_scroll(res)

            return res, scroll_id
        else:
            pass


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
        user_inputs = json.loads(request_data.get("data")).get("user_inputs")

        search_helper = Search(user_query=user_inputs)
        json_data, scroll_id = search_helper.search()

        data_cards = []

        for x in json_data['hits']['hits']:
            json_payload = {}
            for key, value in x['_source'].items(): json_payload[key] = value
            json_payload['_score'] = x['_score']
            data_cards.append(json_payload)

        total_hits = 0

        try:
            total_hits = data['hits']['total']['value']
        except Exception as e:
            pass

        return {"data": data_cards, "total_hits": total_hits, "scroll_id": scroll_id}

    except Exception as e:
        print("Error", e)
        return "error"


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True)
