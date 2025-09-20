'''
This file takes two paths: the path to a directory containing the knn index files and a path to ground truth index file.
Currently, this program assumes the following things:
    - knn index file
        - Each line in contains the corresponding source point ID followed by the IDs of its k nearest neighbors.
    - The ground truth index file
        - Single file
        - In the first half, contains the IDs of the k nearest neighbors of each point in the dataset
        - The second half contains the distances of the k nearest neighbors of each point in the dataset, which is ignored by this program, if contains-distances is not set.
        - Rows are sorted by point ID.
        - The first row contains the IDs of the k nearest neighbors of point 0, the second row contains the IDs of the k nearest neighbors of point 1, and so on.
        - Each line is sorted by distance.

'''

import argparse
import pathlib
import time
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate kNN index')

    parser.add_argument('-i', '--index',
                        dest='index_file_path',
                        required=True, action='store',
                        help='Path to a directory containing the kNN index files')
    parser.add_argument('-g', '--ground-truth',
                        dest='ground_truth_file_path',
                        required=True, action='store',
                        help='Path to the ground truth index file')
    # k: the number of nearest neighbors to be evaluated
    parser.add_argument('-k', '--k',
                        dest='k',
                        required=True, action='store', type=int,
                        help='The number of nearest neighbors to be evaluated')

    parser.add_argument('-D', '--contains-distances',
                        dest='contains_distances',
                        required=False, action='store_true',
                        help='The ground truth file contains distances')

    args = parser.parse_args()
    return args


def find_files(path):
    if not os.path.isdir(path):
        return [pathlib.Path(path)]
    return [pathlib.Path(path) / f for f in os.listdir(path)]


def read_ground_truth(ground_truth_file_paths, k, contains_distances=False):
    for gt_file_path in ground_truth_file_paths:
        print(f'Open {gt_file_path}')

        num_lines = sum(1 for _ in open(gt_file_path))
        print('Number of lines = ' + str(num_lines))

        if contains_distances:
            num_lines //= 2
        print('Number of points = ' + str(num_lines))

        gt_data = []
        with open(gt_file_path, 'r') as f:
            for i in range(num_lines):
                nn = list(map(int, f.readline().split()))
                if len(nn) < k:
                    print(
                        f'Warning: the number of neighbors of point {i} in the ground truth is less than {k}, which is {len(nn)}')
                    exit(1)

                nn = nn[:k]
                nn_set = set(nn)
                assert len(nn_set) == len(nn)
                gt_data.append(nn_set)

    print('Number of points in ground truth = ' + str(len(gt_data)))
    return gt_data


def evaluate(index_file_path, ground_truth, k):
    num_corrects = 0
    num_lines = 0
    print(f'Open {index_file_path}')
    with open(index_file_path, 'r') as f:
        for line in f:
            [source_point_id, *knn_ids] = map(int, line.split())

            if k > len(knn_ids):
                print(
                    f'Warning: the number of neighbors of point {source_point_id} in the kNN index file is less than {k}, which is {len(knn_ids)}')
                exit(1)

            # reduce the number of neighbors to k
            knn_ids = knn_ids[:k]
            knn_ids_set = set(knn_ids)

            # Make sure that the kNN index file contains unique neighbors
            if len(knn_ids_set) != len(knn_ids):
                print(
                    f'Warning: There are duplicate neighbors in point {source_point_id}')
                exit(1)

            ground_truth_knn_ids = ground_truth[source_point_id]
            if len(ground_truth_knn_ids) < k:
                print(
                    f'Warning: the number of neighbors of point {source_point_id} in the ground truth is less than {k}, which is {len(ground_truth_knn_ids)}')
                exit(1)

            num_corrects += len(knn_ids_set & ground_truth_knn_ids)
            num_lines += 1

    return num_corrects, num_lines


def main():
    args = parse_args()
    print(args)

    print('Read ground truth')
    ground_truth = read_ground_truth(
        find_files(args.ground_truth_file_path), args.k,
        args.contains_distances)

    index_files = find_files(args.index_file_path)
    print('Found ' + str(len(index_files)) + ' index files')

    num_corrects = 0
    num_neighbors = 0
    num_lines = 0
    start_time = time.time()
    for index_file in index_files:
        corrects, lines = evaluate(index_file, ground_truth, args.k)
        num_corrects += corrects
        num_lines += lines
        num_neighbors += lines * args.k
    end_time = time.time()

    print(f'Number of points in index = {num_lines}')
    print(f'Number of neighbors = {num_neighbors}')
    print(f'Number of correct neighbors = {num_corrects}')
    print(f'Recall Score = {num_corrects / num_neighbors}')
    print(f'Time = {end_time - start_time}')

    assert num_lines == len(ground_truth)


if __name__ == '__main__':
    main()
