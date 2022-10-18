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
        return hdf5_file['train'], hdf5_file['test'], hdf5_file['neighbors'], hdf5_file.attrs['distance']


def write_train(data, out_file_prefix):
    with open(out_file_prefix + '-train.txt', 'w') as f:
        for point in data:
            for i, feature in enumerate(point):
                if i > 0:
                    f.write('\t')
                f.write(feature)
            f.write('\n')


def write_test(data, out_file_prefix):
    with open(out_file_prefix + '-test.txt', 'w') as f:
        for point in data:
            for i, feature in enumerate(point):
                if i > 0:
                    f.write('\t')
                f.write(feature)
            f.write('\n')


def write_ground_truth(neighbors, distance, out_file_prefix):
    with open(out_file_prefix + '-gt.txt', 'w') as f:
        for entry in neighbors:
            for i, id in enumerate(entry):
                if i > 0:
                    f.write('\t')
                f.write(id)
            f.write('\n')

        for entry in distance:
            for i, d in enumerate(entry):
                if i > 0:
                    f.write('\t')
                f.write(d)
            f.write('\n')


def main(argv):
    option = parse_options(argv)

    print ("Read data from : ", option.input_dataset_path)
    train, test, neighbors, distance = parse_options(option.input_dataset_path)

    write_train(train, option.output_prefix)
    write_test(test, option.output_prefix)
    write_ground_truth(neighbors, distance, option.output_prefix)

    print('Finished conversion')


if __name__ == '__main__':
    main(sys.argv[1:])