from process_dataset import process_single_with_glove_vectors
from analyze_datasets import analyze
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create Wikipedia node classification dataset from sources')
    parser.add_argument('--glove-embedding-file', help='Word embedding file')
    parser.add_argument('--output-dir', help='Directory to write results to')
    parser.add_argument('--batch-dir', help='Directory to write results to')

    args = parser.parse_args()
    for path, _subdirs, files in os.walk(args.batch_dir):
        for name in files:
            if name != 'node.txt': continue
            process_single_with_glove_vectors(path, args.glove_embedding_file)
            
    # process_single_with_glove_vectors(args.output_dir, args.glove_embedding_file)
    # analyze(args.output_dir)
    # python data_processing/pyscripts/node_embedder.py --glove-embedding-file dataset/embeddings/glove.42B.300d.txt --output-dir dataset/test/cross_platform
    # python data_processing/pyscripts/node_embedder.py --glove-embedding-file dataset/glove_embedding.txt --output-dir dataset/test/cross_platform