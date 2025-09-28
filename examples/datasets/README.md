## 5-4 Dataset Overview

The 5-4 dataset is a collection of data files related to a set of **5**
clusters (groups of close points),
each containing **4** points.
The dataset consists of the following files:

- [point_5-4.txt](./point_5-4.txt):
    - Contains 20 feature vectors (5 dimensions).
    - White-space separated value (WSV) format.
    - Distance metric is L2 (Euclidean) distance.

- [point_5-4_id.txt](./point_5-4_id.txt):
    - Contains the same data as point_5-4.txt.
    - Each line starts with an ID, followed by the feature vector.
    - WSV-ID format.
    - IDs are not sequential, but unique.
    - The digit in the tens place of the ID indicates the cluster to which the
      point belongs.

- [query_5-4.txt](./query_5-4.txt):
    - Consists of 5 search queries.
    - Each query point is close to one of the clusters in the point_5-4.txt
      file.
    - There is a one-to-one correspondence between the query points and the
      clusters.

- [ground-truth_5-4.txt](./ground-truth_5-4.txt):
    - Contains ground truth data of the 5 queries.
    - The first half of the file lists the ground truth nearest neighbor IDs.
    - The second half of the file lists the ground truth distances.
    - For example, the first line is for the ground truth nearest neighbor IDs
      of the first query point. The sixth line contains the ground truth
      distances of the first query point.

- [all-distance-pairs_5-4.txt](./all-distance-pairs_5-4.txt):
    - This file contains all possible distance pairs between the input points in
      the dataset.

## String Dataset

There is also a string dataset.

- [point_string.txt](./point_string.txt):
    - Contains 9 strings with different lengths.
    - Distance function is the Levenshtein.

- [query_string.txt](./query_string.txt):
    - Contains 5 queries.

- [ground-truth_string.txt](./ground-truth_string.txt):
    - Contains the ground truth data of the 5 queries.
    - The same format as the ground-truth_5-4.txt file.
    - For each query, all data point IDs and distances to them from the query
      point are listed, sorted by the distance.

## Fashion-MNIST

There is a subset of
the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).

- [fashion-mnist_200.txt](./fashion-mnist_200.txt):
    - Contains 200 data points from the Fashion-MNIST dataset.
    - WSV format.

- [fashion-mnist_200_ground-truth_knng_k10.txt](./fashion-mnist_200_ground-truth_knng_k20.txt):
    - Contains the ground truth k-nearest neighbors (k-NN) graph for the 200
      data points.
    - L2 (Euclidean) distance metric was used.
    - The format is as follows:
        - The first half of each line contains the IDs of the 10 nearest
          neighbors.
        - The second half contains the corresponding distances to those
          neighbors.
        - Line 0 corresponds to the data point with ID 0, line 1 to ID 1, and so
          on.
        - Line 201 contains the distance values of the 10 nearest neighbors for
          the data point with ID 0, line 202 for ID 1, and so on.