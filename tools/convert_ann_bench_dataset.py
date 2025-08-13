'''
Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
saltatlas Project Developers. See the top-level COPYRIGHT file for details.

SPDX-License-Identifier: MIT
'''

'''
Convert the original (HDF5 format) ANN-Benchmarks dataset to text.
Original datasets are available at https://github.com/erikbern/ann-benchmarks/.

This program generates three separated files for
Train (input points), test (query), and ground truth.

[Setup]
python3 -m pip install h5py

[Usage]
python3 convert_ann_bench_dataset.py -h

'''

import sys
import argparse
import h5py
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='ANN-Benchmarks dataset converter')

    parser.add_argument('-i', '--input',
                        dest='input_file_path',
                        required=True, action='store',
                        help='Path to an input (original) HDF5 file')

    parser.add_argument('-o', '--output',
                        dest='output_file_prefix',
                        required=True, action='store',
                        help='Prefix of output file(s)')

    args = parser.parse_args()
    return args


def read_data(dataset_path):
    with h5py.File(dataset_path, "r") as hdf5_file:
        print('Open ' + dataset_path)
        print(hdf5_file['train'])
        print(hdf5_file['test'])
        print(hdf5_file['neighbors'])
        print(hdf5_file['distances'])
        print('Distance =\t' + hdf5_file.attrs['distance'])
        return np.array(hdf5_file['train']), np.array(
            hdf5_file['test']), np.array(hdf5_file['neighbors']), np.array(
            hdf5_file['distances'])


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
    if data.dtype == bool:
        print('Convert bool to int')
        data = data.astype(int)
        print(data)

    with open(out_file_prefix + '-query.txt', 'w') as f:
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


def main():
    arg = parse_args()

    print("Read data from : ", arg.input_file_path)
    train, test, neighbors, distances = read_data(arg.input_file_path)

    write_train(train, arg.output_file_prefix)
    write_test(test, arg.output_file_prefix)
    write_ground_truth(neighbors, distances, arg.output_file_prefix)

    print('Finished conversion')


if __name__ == '__main__':
    main()
