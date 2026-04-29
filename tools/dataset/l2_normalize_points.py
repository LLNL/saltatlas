import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='L2-normalize points and write them back in text format.')
    parser.add_argument('-i', '--input',
                        dest='input_file_path',
                        required=True, action='store',
                        help='Path to an input points file')
    parser.add_argument('-o', '--output',
                        dest='output_file_path',
                        required=True, action='store',
                        help='Path to an output points file')
    parser.add_argument('-t', '--type',
                        dest='file_format',
                        default='tsv', choices=['tsv', 'wsv'],
                        help='File format (tsv or wsv)')
    return parser.parse_args()


def infer_num_dimensions_tsv(input_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line and not line.isspace():
                return len(line.split())
    return 0


def read_points_tsv(input_file_path):
    dim = infer_num_dimensions_tsv(input_file_path)
    if dim == 0:
        return np.empty((0, 0), dtype=np.float32)

    data = np.fromfile(input_file_path, dtype=np.float32, sep=' ')
    if data.size % dim != 0:
        raise ValueError(
            f'Input shape is invalid for TSV: {input_file_path} (dim={dim}, '
            f'values={data.size})')
    return data.reshape(-1, dim)


def l2_normalize_rows(points):
    if points.size == 0:
        return points
    norms = np.sqrt(np.sum(points * points, axis=1, dtype=np.float32))
    nonzero = norms > 0.0
    points[nonzero] /= norms[nonzero, np.newaxis]
    return points


def write_points_tsv(output_file_path, points):
    np.savetxt(output_file_path, points, delimiter=' ', fmt='%.9g')


def main():
    arg = parse_args()

    if arg.file_format != 'tsv' and arg.file_format != 'wsv':
        raise ValueError(f'Unsupported file format: {arg.file_format}')

    points = read_points_tsv(arg.input_file_path)
    points = l2_normalize_rows(points)
    write_points_tsv(arg.output_file_path, points)


if __name__ == '__main__':
    main()
