import argparse
import pathlib
import time
import os

"""Evaluate kNN index files against a ground-truth.

This script reads k-nearest-neighbor index files and compares them
to ground-truth neighbor lists. Supported file base formats are:

- `N`: single-line neighbor lists (optionally prefixed by an ID)
- `H`: two-line human-readable format where the first line lists IDs
    and the second line lists distances
- `M`: a two-line non-interleaved format (treated like `H` by this
    tool)

Optionally, files can include an `I` flag to indicate that each line
contains an explicit source ID (e.g. `NI`, `HI`, `MI`). By default
index files are interpreted as `NI` and ground-truth as `N`.

Usage examples:

  python tools/evalulate_knng.py -i index.txt -g groundtruth.txt -k 10
  python tools/evalulate_knng.py -i index_dir -g gt_dir -k 20 -I NI -G NI

Run with `-D` to enable distance-based evaluation (requires distance
arrays to be present in the two-line formats).
"""

FORMAT_BASES = {'H', 'M', 'N'}


def parse_args():
    """Parse command-line arguments.

    Returns a namespace with parsed arguments. See the module-level
    docstring for usage examples and format explanations.
    """
    parser = argparse.ArgumentParser(
        description='Evaluate kNN index')

    parser.add_argument('-i', '--index',
                        dest='index_file_path',
                        required=True, action='store',
                        help='Path to a kNN index file or a directory containing the kNN index files')
    parser.add_argument('-g', '--ground-truth',
                        dest='ground_truth_file_path',
                        required=True, action='store',
                        help='Path to the ground truth index file')
    # k: the number of nearest neighbors to be evaluated
    parser.add_argument('-k', '--k',
                        dest='k',
                        required=True, action='store', type=int,
                        help='The number of nearest neighbors to be evaluated')

    parser.add_argument('-I', '--index-format',
                        dest='index_format',
                        required=False, action='store',
                        help='Format of the kNN index files: H/M/N plus optional I (default: NI)')
    parser.add_argument('-G', '--ground-truth-format',
                        dest='ground_truth_format',
                        required=False, action='store',
                        help='Format of the ground truth files: H/M/N plus optional I (default: N)')

    parser.add_argument('-D', '--dist-eval',
                        dest='use_distance_eval',
                        required=False, action='store_true',
                        help='Enable distance-based evaluation: a neighbor is correct if its distance is <= ground-truth k-th neighbor distance')

    args = parser.parse_args()
    if args.index_format is None:
        args.index_format = 'NI'
    if args.ground_truth_format is None:
        args.ground_truth_format = 'N'
    try:
        args.index_format = parse_format(args.index_format)
        args.ground_truth_format = parse_format(args.ground_truth_format)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def find_files(path):
    """Return a list of pathlib.Path objects for a file or directory.

    If `path` is a file path, return a single-element list. If it's a
    directory, return a sorted list of entries inside it.
    """
    if not os.path.isdir(path):
        return [pathlib.Path(path)]
    return sorted(pathlib.Path(path) / f for f in os.listdir(path))


def parse_format(fmt):
    """Parse a short format string into a dict.

    The format string must include one of the base letters `H`, `M`, or
    `N`. Optionally include `I` to indicate per-line IDs (e.g. "NI").
    Returns a dict with keys: `base`, `has_id`, and `spec`.
    """
    if fmt is None:
        raise ValueError('Format is missing')
    s = fmt.strip().upper()
    if not s:
        raise ValueError('Format is empty')
    has_id = 'I' in s
    base = None
    for ch in s:
        if ch == 'I':
            continue
        if ch in FORMAT_BASES:
            if base is not None and base != ch:
                raise ValueError(f'Format has multiple base letters: {fmt}')
            base = ch
        else:
            raise ValueError(f'Invalid format letter: {ch}')
    if base is None:
        raise ValueError(f'Format must include one of {sorted(FORMAT_BASES)}')
    return {'base': base, 'has_id': has_id, 'spec': s}

def count_lines(path):
    with open(path, 'r') as f:
        return sum(1 for _ in f)


def token_has_float(token):
    return '.' in token or 'e' in token or 'E' in token


def has_float_tokens(tokens):
    for token in tokens:
        if token_has_float(token):
            return True
    return False


def can_parse_first_k_as_int(tokens, k):
    if len(tokens) < k:
        return False
    try:
        _ = [int(x) for x in tokens[:k]]
        return True
    except Exception:
        return False


def iter_id_rows(file_path, file_format, start_id=0):
    """Yield (source_id, token_list) for supported formats.

    This generator handles the three base formats and respects whether
    each line contains an explicit source ID. `start_id` is used when
    IDs are implicit (auto-incremented across files).
    """
    base = file_format['base']
    has_id = file_format['has_id']

    if base == 'N':
        if has_id:
            with open(file_path, 'r') as f:
                for line in f:
                    tokens = line.split()
                    yield int(tokens[0]), tokens[1:]
            return
        with open(file_path, 'r') as f:
            sid = start_id
            for line in f:
                yield sid, line.split()
                sid += 1
        return

    if base == 'H':
        # Support both layouts:
        # 1) interleaved: [ids][dists][ids][dists]...
        # 2) blocked:     [all ids...][all dists...]
        with open(file_path, 'r') as f:
            rows = [line.split() for line in f]
        num_lines = len(rows)
        num_points = num_lines // 2
        if num_lines % 2 != 0:
            raise ValueError(f'H format expects even number of lines, got {num_lines} in {file_path}')

        interleaved = True
        if num_lines >= 2 and not has_float_tokens(rows[1]):
            interleaved = False

        if interleaved:
            for i in range(num_points):
                id_tokens = rows[2 * i]
                if has_id:
                    yield int(id_tokens[0]), id_tokens[1:]
                else:
                    yield start_id + i, id_tokens
        else:
            for i in range(num_points):
                id_tokens = rows[i]
                if has_id:
                    yield int(id_tokens[0]), id_tokens[1:]
                else:
                    yield start_id + i, id_tokens
        return

    if base == 'M':
        # Always treat the 'M' base as the non-interleaved two-line format.
        # Distances may not be simple floats, so detecting interleaved lines
        # by scanning for '.'/'e' is unreliable; therefore we always follow
        # the original non-interleaved parsing logic.
        with open(file_path, 'r') as f:
            first_line = f.readline()
            if not first_line:
                return
            first_tokens = first_line.split()
            if has_id:
                # Each point is represented by an ID line followed by an
                # auxiliary line (e.g., distances). We yield the ID and the
                # token list that follows the ID on the ID line.
                yield int(first_tokens[0]), first_tokens[1:]
                _ = f.readline()
                while True:
                    id_line = f.readline()
                    if not id_line:
                        break
                    tokens = id_line.split()
                    yield int(tokens[0]), tokens[1:]
                    _ = f.readline()
            else:
                # No per-line ID: each point occupies two lines; the first
                # line for the first point has already been read.
                sid = start_id
                yield sid, first_tokens
                sid += 1
                _ = f.readline()
                while True:
                    id_line = f.readline()
                    if not id_line:
                        break
                    tokens = id_line.split()
                    yield sid, tokens
                    sid += 1
                    _ = f.readline()
        return

    raise ValueError(f'Unknown format: {file_format}')


def iter_id_rows_with_dists(file_path, file_format, start_id=0):
    """Yield (source_id, id_tokens_list, dist_tokens_list_or_None).

    This supports the same formats as `iter_id_rows` but attempts to
    return distances when the format uses a two-line representation
    (e.g., base 'H' or our non-interleaved 'M'). When distances are not
    present for a format, the third element will be None.
    """
    base = file_format['base']
    has_id = file_format['has_id']

    if base == 'N':
        # Single-line neighbor lists: no distances available
        if has_id:
            with open(file_path, 'r') as f:
                for line in f:
                    tokens = line.split()
                    yield int(tokens[0]), tokens[1:], None
            return
        with open(file_path, 'r') as f:
            sid = start_id
            for line in f:
                yield sid, line.split(), None
                sid += 1
        return

    if base == 'H':
        # Support both layouts:
        # 1) interleaved: [ids][dists][ids][dists]...
        # 2) blocked:     [all ids...][all dists...]
        with open(file_path, 'r') as f:
            rows = [line.split() for line in f]
        num_lines = len(rows)
        num_points = num_lines // 2
        if num_lines % 2 != 0:
            raise ValueError(f'H format expects even number of lines, got {num_lines} in {file_path}')

        interleaved = True
        if num_lines >= 2 and not has_float_tokens(rows[1]):
            interleaved = False

        if interleaved:
            for i in range(num_points):
                id_tokens = rows[2 * i]
                dist_tokens = rows[2 * i + 1]
                if has_id:
                    yield int(id_tokens[0]), id_tokens[1:], dist_tokens
                else:
                    yield start_id + i, id_tokens, dist_tokens
        else:
            for i in range(num_points):
                id_tokens = rows[i]
                dist_tokens = rows[num_points + i]
                if has_id:
                    yield int(id_tokens[0]), id_tokens[1:], dist_tokens
                else:
                    yield start_id + i, id_tokens, dist_tokens
        return

    if base == 'M':
        # We always parse the non-interleaved two-line form: an ID/ids line
        # followed by an auxiliary distances line.
        with open(file_path, 'r') as f:
            first_line = f.readline()
            if not first_line:
                return
            first_tokens = first_line.split()
            if has_id:
                # read the following distances line
                dist_line = f.readline()
                dist_tokens = dist_line.split() if dist_line else []
                yield int(first_tokens[0]), first_tokens[1:], dist_tokens
                while True:
                    id_line = f.readline()
                    if not id_line:
                        break
                    tokens = id_line.split()
                    dist_line = f.readline()
                    dist_tokens = dist_line.split() if dist_line else []
                    yield int(tokens[0]), tokens[1:], dist_tokens
            else:
                sid = start_id
                dist_line = f.readline()
                dist_tokens = dist_line.split() if dist_line else []
                yield sid, first_tokens, dist_tokens
                sid += 1
                while True:
                    id_line = f.readline()
                    if not id_line:
                        break
                    tokens = id_line.split()
                    dist_line = f.readline()
                    dist_tokens = dist_line.split() if dist_line else []
                    yield sid, tokens, dist_tokens
                    sid += 1
        return

    raise ValueError(f'Unknown format: {file_format}')


def read_ground_truth(ground_truth_file_paths, k, ground_truth_format, use_distance_eval=False):
    # Return a tuple: (gt_id_dict, gt_kth_dist_dict_or_None)
    """Read ground-truth neighbor lists from one or more files.

    Returns a tuple `(gt_id_dict, gt_kth_dist_dict_or_None)`. If
    `use_distance_eval` is True, the second element is a dict mapping
    source IDs to the k-th neighbor distance; otherwise it is None.
    The function validates neighbor count and uniqueness and will
    exit(1) on malformed input to make issues explicit to users.
    """
    gt_data = {}
    gt_kth_dist = {} if use_distance_eval else None
    implicit_ids = not ground_truth_format['has_id']
    next_auto_id = 0
    for gt_file_path in ground_truth_file_paths:
        print(f'Open {gt_file_path}')
        if ground_truth_format['base'] == 'H':
            num_lines = count_lines(gt_file_path)
            print('Number of lines = ' + str(num_lines))
            print('Number of points = ' + str(num_lines // 2))

        num_points = 0
        if use_distance_eval:
            iterator = iter_id_rows_with_dists
        else:
            iterator = iter_id_rows

        for parsed in iterator(gt_file_path, ground_truth_format, start_id=next_auto_id):
            if use_distance_eval:
                source_point_id, nn_tokens, dist_tokens = parsed
                # Some datasets store two-line records as [distances][ids]
                # instead of [ids][distances]. In distance-eval mode, detect
                # and auto-correct this to avoid hard failures.
                if dist_tokens is not None:
                    ids_look_valid = can_parse_first_k_as_int(nn_tokens, k)
                    dists_look_like_ids = can_parse_first_k_as_int(dist_tokens, k)
                    if (not ids_look_valid) and dists_look_like_ids:
                        print(
                            f'Warning: swapped ID/distance rows detected for point {source_point_id} in ground truth; auto-correcting')
                        nn_tokens, dist_tokens = dist_tokens, nn_tokens
            else:
                source_point_id, nn_tokens = parsed

            if len(nn_tokens) < k:
                print(
                    f'Warning: the number of neighbors of point {source_point_id} in the ground truth is less than {k}, which is {len(nn_tokens)}')
                exit(1)

            try:
                nn_ids = [int(x) for x in nn_tokens[:k]]
            except Exception:
                print(
                    f'Error: cannot parse ground truth neighbor IDs for point {source_point_id}. '
                    f'Check --ground-truth-format (current: {ground_truth_format["spec"]}).')
                exit(1)
            nn_set = set(nn_ids)
            if len(nn_set) != len(nn_ids):
                print(
                    f'Warning: There are duplicate neighbors in point {source_point_id} in the ground truth')
                exit(1)

            if implicit_ids:
                gt_data[next_auto_id + num_points] = nn_set
            else:
                if source_point_id in gt_data:
                    print(
                        f'Warning: duplicate source point ID {source_point_id} in the ground truth')
                    exit(1)
                gt_data[source_point_id] = nn_set

            if use_distance_eval:
                # extract k-th neighbor distance if available
                if not dist_tokens or len(dist_tokens) < k:
                    print(
                        f'Error: ground truth distances missing or shorter than k for point {source_point_id}')
                    exit(1)
                try:
                    kth_dist = float(dist_tokens[k - 1])
                except Exception:
                    print(f'Error: cannot parse ground truth distance for point {source_point_id}')
                    exit(1)
                gt_kth_dist[source_point_id] = kth_dist

            num_points += 1

        if implicit_ids:
            next_auto_id += num_points

    if not implicit_ids:
        if len(gt_data) == 0:
            print('Warning: ground truth is empty')
            exit(1)
        min_id = min(gt_data.keys())
        max_id = max(gt_data.keys())
        if min_id != 0:
            print(f'Warning: missing ground truth entry for point 0')
            exit(1)
        for i in range(0, max_id + 1):
            if i not in gt_data:
                print(f'Warning: missing ground truth entry for point {i}')
                exit(1)

    print('Number of points in ground truth = ' + str(len(gt_data)))
    return gt_data, gt_kth_dist


def evaluate(index_file_path, ground_truth_ids, ground_truth_kth_dist, k, index_format, start_id=0):
    """Evaluate a single index file against prepared ground truth.

    Returns a tuple `(num_corrects, num_lines, next_start_id)`. If
    `ground_truth_kth_dist` is None an ID-based comparison is used;
    otherwise distance-based evaluation is performed and distances are
    parsed from the index file.
    """
    num_corrects = 0
    num_lines = 0
    print(f'Open {index_file_path}')
    implicit_ids = not index_format['has_id']

    if ground_truth_kth_dist is None:
        # ID-based evaluation (existing behavior)
        for source_point_id, knn_tokens in iter_id_rows(
                index_file_path, index_format, start_id=start_id):
            if len(knn_tokens) < k:
                print(
                    f'Warning: the number of neighbors of point {source_point_id} in the kNN index file is less than {k}, which is {len(knn_tokens)}')
                exit(1)

            knn_ids = [int(x) for x in knn_tokens[:k]]
            knn_ids_set = set(knn_ids)

            # Make sure that the kNN index file contains unique neighbors
            if len(knn_ids_set) != len(knn_ids):
                print(
                    f'Warning: There are duplicate neighbors in point {source_point_id}')
                exit(1)

            if source_point_id not in ground_truth_ids:
                print(
                    f'Warning: missing ground truth for point {source_point_id}')
                exit(1)

            ground_truth_knn_ids = ground_truth_ids[source_point_id]
            if len(ground_truth_knn_ids) < k:
                print(
                    f'Warning: the number of neighbors of point {source_point_id} in the ground truth is less than {k}, which is {len(ground_truth_knn_ids)}')
                exit(1)

            num_corrects += len(knn_ids_set & ground_truth_knn_ids)
            num_lines += 1
    else:
        # Distance-based evaluation: do not check IDs. A neighbor is correct if
        # its distance <= ground-truth k-th neighbor distance for that source.
        for source_point_id, knn_ids, knn_dists in iter_id_rows_with_dists(
                index_file_path, index_format, start_id=start_id):
            if len(knn_ids) < k:
                print(
                    f'Warning: the number of neighbors of point {source_point_id} in the kNN index file is less than {k}, which is {len(knn_ids)}')
                exit(1)

            if knn_dists is None or len(knn_dists) < k:
                print(f'Error: index file does not contain distances required for distance-based evaluation for point {source_point_id}')
                exit(1)

            if source_point_id not in ground_truth_kth_dist:
                print(
                    f'Warning: missing ground truth distance for point {source_point_id}')
                exit(1)

            try:
                threshold = float(ground_truth_kth_dist[source_point_id])
            except Exception:
                print(f'Error: cannot parse ground truth k-th distance for point {source_point_id}')
                exit(1)

            # Count neighbors whose distance is <= threshold
            corrected = 0
            for d in knn_dists[:k]:
                try:
                    dv = float(d)
                except Exception:
                    print(f'Error: cannot parse index neighbor distance for point {source_point_id}')
                    exit(1)
                if dv <= threshold:
                    corrected += 1
            num_corrects += corrected
            num_lines += 1

    if implicit_ids:
        start_id += num_lines

    return num_corrects, num_lines, start_id


def main():
    """Main entry point: parse args, read ground truth, and evaluate.

    Prints a short summary including recall and timing information.
    """
    args = parse_args()
    print('Arguments:')
    print(args)

    print('Read ground truth')
    gt_files = find_files(args.ground_truth_file_path)
    if len(gt_files) > 1 and not args.ground_truth_format['has_id']:
        print('Error: multiple ground truth files require IDs in the file format')
        exit(1)
    ground_truth_ids, ground_truth_kth_dist = read_ground_truth(
        gt_files, args.k, args.ground_truth_format, use_distance_eval=args.use_distance_eval)

    index_files = find_files(args.index_file_path)
    if len(index_files) > 1 and not args.index_format['has_id']:
        print('Error: multiple index files require IDs in the file format')
        exit(1)
    print('Found ' + str(len(index_files)) + ' index files')

    num_corrects = 0
    num_neighbors = 0
    num_lines = 0
    start_time = time.time()
    next_auto_id = 0
    implicit_ids = not args.index_format['has_id']
    for index_file in index_files:
        corrects, lines, next_auto_id = evaluate(
            index_file, ground_truth_ids, ground_truth_kth_dist, args.k, args.index_format, start_id=next_auto_id)
        num_corrects += corrects
        num_lines += lines
        num_neighbors += lines * args.k
        if not implicit_ids:
            next_auto_id = 0
    end_time = time.time()

    print(f'Number of points in index = {num_lines}')
    print(f'Number of neighbors = {num_neighbors}')
    print(f'Number of correct neighbors = {num_corrects}')
    print(f'Recall Score = {num_corrects / num_neighbors}')
    print(f'Time = {end_time - start_time}')

    assert num_lines == len(ground_truth_ids)


if __name__ == '__main__':
    main()
