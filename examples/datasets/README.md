## 5-4 Dataset Overview

The 5-4 dataset is a collection of data files related to a set of **5** clusters (groups of close points),
each containing **4** points.
The dataset consists of the following files:

- [point_5-4.txt](./point_5-4.txt):
  - Contains 20 feature vectors (5 dimensions). 
  - White-space separated value (WSV) format.
  - Distance metric is L2 (Euclidean) distance.

- [query_5-4.txt](./query_5-4.txt):
  - Consists of 5 search queries.
  - Each query point is close to one of the clusters in the point_5-4.txt file.
  - There is a one-to-one correspondence between the query points and the clusters.

- [ground-truth_5-4.txt](./ground-truth_5-4.txt):
  - Contains ground truth data of the 5 queries.
  - The first half of the file lists the ground truth nearest neighbor IDs.
  - The second half of the file lists the ground truth distances.
  - For example, the first line is for the ground truth nearest neighbor IDs of the first query point. The sixth line contains the ground truth distances of the first query point.

- [all-distance-pairs_5-4.txt](./all-distance-pairs_5-4.txt):
  - This file contains all possible distance pairs between the input points in the dataset.

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
  - For each query, all data point IDs and distances to them from the query point are listed, sorted by the distance.