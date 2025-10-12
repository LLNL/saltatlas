'''
Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
saltatlas Project Developers. See the top-level COPYRIGHT file for details.

SPDX-License-Identifier: MIT
'''

'''
Convert the original (binary) Big ANN Benchmark dataset to text.
Original datasets are available at https://big-ann-benchmarks.com/.

This program converts three types of datasets:
base (points), query, and ground truth.

Base datasets are split into multiple files (1 file per thread)
Query and ground truth datasets are converted to a single file for each.

[Usage]
python3 ./convert_big_ann_bench_dataset.py -h
'''

import os
import sys
import argparse
import pathlib
import numpy as np
import threading
import multiprocessing


def parse_args():
    parser = argparse.ArgumentParser(
        description='Big ANN Benchmark dataset converter')

    parser.add_argument('-i', '--input',
                        dest='input_file_path',
                        required=True, action='store',
                        help='Path to an input (original) file')
    parser.add_argument('-o', '--output',
                        dest='output_file_prefix',
                        required=True, action='store',
                        help='Prefix of output file(s)')
    parser.add_argument('-b', '--base', dest='dataset_type',
                        action='store_const', const='base',
                        help='Parse base dataset')
    parser.add_argument('-q', '--query', dest='dataset_type',
                        action='store_const', const='query',
                        help='Parse query dataset')
    parser.add_argument('-g', '--ground-truth', dest='dataset_type',
                        action='store_const', const='gt',
                        help='Parse ground truth dataset')
    parser.add_argument('-t', type=int, dest='num_threads',
                        action='store', default='0',
                        help='Number of threads to use (if 0 is specified, use all available threads)')

    args = parser.parse_args()
    return args


def get_data_type(file_path):
    ext = pathlib.Path(file_path).suffix

    if ext == '.u8bin':
        return np.uint8
    elif ext == '.i8bin':
        return np.int8
    elif ext == '.fbin':
        return np.float32
    else:
        print('Unexpected file extension: ' % file_path, file=sys.stderr)
        exit(1)
        return None


def numpy_datatype_to_formatting_types(datatype):
    if datatype == np.uint8:
        return '%d'
    elif datatype == np.int8:
        return '%d'
    elif datatype == np.float32:
        return '%f'
    else:
        assert False, 'Unsupported data type'


# Performance optimized numpy.savetxt()
def save_numpy_2d_array(output_file_path, append, array, data_element_format):
    with open(output_file_path, 'a' if append else 'w') as f:
        fmt = ' '.join([data_element_format] * array.shape[1])
        fmt = '\n'.join([fmt] * array.shape[0])
        data = fmt % tuple(array.ravel())
        f.write(data + '\n')


def convert_points(input_file_path, output_file_path, count, start_idx,
                   chunk_size=None):
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    with open(input_file_path, 'rb') as fi:
        dtype = get_data_type(input_file_path)

        num_points, dim = np.fromfile(fi, count=2, dtype=np.int32)
        fi.seek(start_idx * dim * dtype().itemsize, 1)

        if not chunk_size:
            chunk_size = count

        for i in range(0, count, chunk_size):
            n = min(chunk_size, count - i)
            print(f'{(i / count * 100):.2f} %')  # Show progress
            arr = np.fromfile(fi, count=n * dim, dtype=dtype)
            assert arr.size > 0
            arr = arr.reshape(n, dim)
            save_numpy_2d_array(output_file_path, i > 0, arr,
                                numpy_datatype_to_formatting_types(
                                    dtype))


def convert_base_parallel(input_file_path, output_file_prefix, num_threads=1,
                          chunk_size=None):
    print(f'#of thread\t{num_threads}')

    assign_table = []
    with open(input_file_path, 'rb') as f:
        num_points, dim = np.fromfile(f, count=2, dtype=np.int32)
        print(f'#of point\t{num_points}')
        print(f'#of dimensions\t{dim}')

        for t in range(num_threads):
            assign_table.append(num_points // num_threads)
            rem = num_points % num_threads
            if t < rem:
                assign_table[-1] += 1

        assert sum(assign_table) == num_points

    threads = []
    start_idx = 0
    for t in range(num_threads):
        output_file_path = output_file_prefix + '-' + str(t) + '.txt'
        threads.append(threading.Thread(target=convert_points,
                                        args=(
                                            input_file_path, output_file_path,
                                            assign_table[t], start_idx,
                                            chunk_size)))
        threads[-1].start()
        start_idx += assign_table[t]

    for t in threads:
        t.join()


def convert_query(input_file_path, output_file_prefix):
    with open(input_file_path, 'rb') as f:
        num_points, dim = np.fromfile(f, count=2, dtype=np.int32)
        print(f'#of point\t{num_points}')
        print(f'#of dimensions\t{dim}')

        convert_points(input_file_path, output_file_prefix + '.txt', num_points,
                       0)


def convert_ground_truth(input_file_path, output_file_prefix):
    with open(input_file_path, 'rb') as fi:
        num_queries, num_neighbors = np.fromfile(fi, count=2, dtype=np.int32)
        print(f'#of queries\t{num_queries}')
        print(f'#of neighbors\t{num_neighbors}')

        ids = np.fromfile(fi, count=num_queries * num_neighbors,
                          dtype=np.uint32)
        ids = ids.reshape(num_queries, num_neighbors)

        output_file_path = output_file_prefix + '.txt'
        save_numpy_2d_array(output_file_path, False, ids, '%d')

        distances = np.fromfile(fi, count=num_queries * num_neighbors,
                                dtype=np.float32)
        distances = distances.reshape(num_queries, num_neighbors)
        save_numpy_2d_array(output_file_path, True, distances, '%f')


def main():
    arg = parse_args()

    t = arg.num_threads
    if t == 0:
        t = multiprocessing.cpu_count()

    if arg.dataset_type == 'base':
        convert_base_parallel(arg.input_file_path, arg.output_file_prefix,
                              t, 2 ** 16)
    elif arg.dataset_type == 'query':
        convert_query(arg.input_file_path, arg.output_file_prefix)
    elif arg.dataset_type == 'gt':
        convert_ground_truth(arg.input_file_path, arg.output_file_prefix)
    else:
        assert False, 'Unsupported dataset type'

    print('Finished conversion')


if __name__ == '__main__':
    main()
