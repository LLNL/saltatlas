'''
[Setup]
python3 -m pip install h5py

[Run]
python3 convert_ann_bench_dataset.py -i ann-bench/hdf5/fashion-mnist-784-euclidean.hdf5 -o ann-bench/txt/fashion-mnist-784-euclidean
python3 convert_ann_bench_dataset.py -i ann-bench/hdf5/glove-25-angular.hdf5            -o ann-bench/txt/glove-25-angular
python3 convert_ann_bench_dataset.py -i ann-bench/hdf5/glove-50-angular.hdf5            -o ann-bench/txt/glove-50-angular
python3 convert_ann_bench_dataset.py -i ann-bench/hdf5/glove-100-angular.hdf5           -o ann-bench/txt/glove-100-angular
python3 convert_ann_bench_dataset.py -i ann-bench/hdf5/glove-200-angular.hdf5           -o ann-bench/txt/glove-200-angular
python3 convert_ann_bench_dataset.py -i ann-bench/hdf5/kosarak-jaccard.hdf5             -o ann-bench/txt/kosarak-jaccard
python3 convert_ann_bench_dataset.py -i ann-bench/hdf5/mnist-784-euclidean.hdf5         -o ann-bench/txt/mnist-784-euclidean
python3 convert_ann_bench_dataset.py -i ann-bench/hdf5/nytimes-256-angular.hdf5         -o ann-bench/txt/nytimes-256-angular
python3 convert_ann_bench_dataset.py -i ann-bench/hdf5/sift-128-euclidean.hdf5          -o ann-bench/txt/sift-128-euclidean
python3 convert_ann_bench_dataset.py -i ann-bench/hdf5/lastfm-64-dot.hdf5               -o ann-bench/txt/lastfm-64-dot
python3 convert_ann_bench_dataset.py -i ann-bench/hdf5/deep-image-96-angular.hdf5       -o ann-bench/txt/deep-image-96-angular
python3 convert_ann_bench_dataset.py -i ann-bench/hdf5/gist-960-euclidean.hdf5          -o ann-bench/txt/gist-960-euclidean
'''

import sys
import getopt
import h5py
import numpy as np


class Option:
    input_dataset_path = ""
    output_prefix = ""


def parse_options(argv):
    option = Option()

    try:
        opts, args = getopt.getopt(argv, "i:o:")
    except getopt.GetoptError:
        print("Wrong arguments")
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-i':
            option.input_dataset_path = arg
        if opt == '-o':
            option.output_prefix = arg

    return option


def read_data(dataset_path):
    with h5py.File(dataset_path, "r") as hdf5_file:
        print('Open ' + dataset_path)
        print (hdf5_file['train'])
        print (hdf5_file['test'])
        print (hdf5_file['neighbors'])
        print (hdf5_file['distances'])
        print('Distance =\t' + hdf5_file.attrs['distance'])
        return np.array(hdf5_file['train']), np.array(hdf5_file['test']), np.array(hdf5_file['neighbors']), np.array(hdf5_file['distances'])


def write_train(data, out_file_prefix):
    print('Write train')
    print(data.shape)
    print(data)
    if data.dtype == bool:
        print('Convert bool to int')
        data = data.astype(int)
        print(data)
    with open(out_file_prefix + '-train.txt', 'w') as f:
        for point in data:
            for i, feature in enumerate(point):
                if i > 0:
                    f.write('\t')
                f.write(str(feature))
            f.write('\n')


def write_test(data, out_file_prefix):
    print('Write test')
    print(data.shape)
    print(data)
    with open(out_file_prefix + '-test.txt', 'w') as f:
        for point in data:
            for i, feature in enumerate(point):
                if i > 0:
                    f.write('\t')
                f.write(str(feature))
            f.write('\n')


def write_ground_truth(neighbors, distances, out_file_prefix):
    print('Write ground truth')
    print(neighbors.shape)
    print(neighbors)
    print(distances.shape)
    print(distances)
    with open(out_file_prefix + '-gt.txt', 'w') as f:
        for entry in neighbors:
            for i, id in enumerate(entry):
                if i > 0:
                    f.write('\t')
                f.write(str(id))
            f.write('\n')

        for entry in distances:
            for i, d in enumerate(entry):
                if i > 0:
                    f.write('\t')
                f.write(str(d))
            f.write('\n')


def main(argv):
    option = parse_options(argv)

    print ("Read data from : ", option.input_dataset_path)
    train, test, neighbors, distances = read_data(option.input_dataset_path)

    write_train(train, option.output_prefix)
    write_test(test, option.output_prefix)
    write_ground_truth(neighbors, distances, option.output_prefix)

    print('Finished conversion')


if __name__ == '__main__':
    main(sys.argv[1:])