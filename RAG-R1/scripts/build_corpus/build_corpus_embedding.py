import faiss
import pickle
from FlagEmbedding import FlagModel
import os
import argparse
import torch

def load_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        corpus = file.readlines()
        c_len = len(corpus)
        new_corpus = []
        line_num = 0
        for line in corpus:
            line_num += 1
            if line_num % 10000 == 0:
                print(f"Percent: {line_num}/{c_len}")

            title_text = line.split('\t')[1].strip('')
            new_corpus.append(title_text)
    return new_corpus


def process_corpus(file_path, save_path):
    print("Start load corpus")
    corpus = load_corpus(file_path)
    print(f"Load {len(corpus)} from {file_path}.")

    # print corpus
    for sample in corpus[:2]:
        print(sample)
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    # Load model (automatically use GPUs)
    model = FlagModel('../model/BAAI__bge-large-en-v1.5',
                      query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                      use_fp16=False)

    split_num = 10
    print("Start encode")
    for i in range(split_num):
        corpus_tmp = corpus[int(len(corpus) * (i / split_num)): int(len(corpus) * ((i + 1) / split_num))]
        corpus_embeddings = model.encode_corpus(corpus_tmp, batch_size=1024, max_length=300)

        print("Shape of the corpus embeddings:", corpus_embeddings.shape)
        print("Data type of the embeddings:", corpus_embeddings.dtype)

        print(f"Start save {i}")
        save_file = save_path + '/split_' + i + '.pickle'
        with open(save_file, 'ab') as f:
            pickle.dump(corpus_embeddings, f)
        print(f"Save over {i}")
    print("All save finish!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process corpus and save embeddings.')
    parser.add_argument('--file_path', type=str, default='../data/kilt/wiki_kilt_100_really.tsv',
                        help='Path to the input TSV file.')
    parser.add_argument('--save_path', type=str, default='../data/kilt/', help='Path to save the output pickle file.')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use (default: 0 for CPU).')

    args = parser.parse_args()

    if torch.cuda.is_available() and args.gpu_id >= 0:
        torch.cuda.set_device(args.gpu_id)
        print(f"Using GPU: {args.gpu_id}")
    else:
        print("Using CPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    process_corpus(args.file_path, args.save_path)
