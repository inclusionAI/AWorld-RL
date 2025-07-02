from flask import Flask, request, jsonify
# import faiss
import numpy as np
from FlagEmbedding import FlagModel
import sys
import argparse
from typing import List
from tqdm import tqdm


def load_corpus(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        corpus = file.readlines()
        corpus = [line.strip("\n") for line in corpus]
    return corpus


# create Flask
app = Flask(__name__)

def load_docs(corpus, doc_idxs):
    docs = [corpus[int(idx)] for idx in doc_idxs]
    results = [{"id": int(doc.split('\t', 1)[0]), "contents": doc.split('\t', 1)[1]} for doc in docs]
    return results


def batch_search(query_list: List[str], num: int = 3, return_score: bool = False):
    batch_size = 128
    if isinstance(query_list, str):
        query_list = [query_list]

    results = []
    scores = []
    for start_idx in tqdm(range(0, len(query_list), batch_size), desc='Retrieval process: '):
        query_batch = query_list[start_idx:start_idx + batch_size]
        batch_emb = model.encode_queries(query_batch)
        batch_scores, batch_idxs = index.search(batch_emb, k=num)
        batch_scores = batch_scores.tolist()
        batch_idxs = batch_idxs.tolist()

        # load_docs is not vectorized, but is a python list approach
        flat_idxs = sum(batch_idxs, [])
        batch_results = load_docs(corpus, flat_idxs)
        # chunk them back
        batch_results = [batch_results[i * num: (i + 1) * num] for i in range(len(batch_idxs))]

        results.extend(batch_results)
        scores.extend(batch_scores)

        del batch_emb, batch_scores, batch_idxs, query_batch, flat_idxs, batch_results

    if return_score:
        return results, scores
    else:
        return results


@app.route("/retrieve", methods=["POST"])
def retrieve():
    # 从请求中获取查询向量
    data = request.json

    queries = data["queries"]
    topk = data.get("topk", 3)
    return_scores = data.get("return_scores", False)

    results, scores = batch_search(
        query_list=queries,
        num=topk,
        return_score=return_scores
    )

    # Format response
    resp = []
    for i, single_result in enumerate(results):
        if return_scores:
            # If scores are returned, combine them with results
            combined = []
            for doc, score in zip(single_result, scores[i]):
                combined.append({"document": doc, "score": score})
            resp.append(combined)
        else:
            resp.append(single_result)
    return {"result": resp}


if __name__ == "__main__":
    data_type = "kilt"
    port = 8000
    faiss_gpu = True

    model = FlagModel(
        "./model/BAAI__bge-large-en-v1.5",
        query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
        use_fp16=False,
    )

    print("Model loading finished")

    # load corpus
    if data_type == "kilt":
        file_path = "./data/kilt/wiki_kilt_100_really.tsv"
    else:
        kill
        file_path = ""
    corpus = load_corpus(file_path)

    print(f"Corpus loading finished-{len(corpus)}")

    # load index
    if data_type == "kilt":
        index_path = "./data/kilt/enwiki_kilt_all.bin"
    else:
        kill
        index_path = ""
    index = faiss.read_index(index_path)
    if faiss_gpu:
        co = faiss.GpuMultipleClonerOptions()
        co.useFloat16 = True
        co.shard = True
        index = faiss.index_cpu_to_all_gpus(index, co=co)

    print("Index loading finished")

    app.run(host="0.0.0.0", port=port, debug=False)  # port:8000
    print(f"You can start searching at http://127.0.0.1:{port}/retrieve")

