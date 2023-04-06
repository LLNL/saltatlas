## 5-4 Dataset Overview

The 5-4 dataset is a collection of data files related to a set of **5** clusters (groups of close points),
each containing **4** points.
The dataset consists of the following files:

- [point_5-4.txt](./point_5-4.txt):
  - This file contains 20 feature vectors (5 dimensions). 
  - The file is in the white-space separated value (WSV) format.
  - The distance metric is L2 (Euclidean) distance.

- [query_5-4.txt](./query_5-4.txt):
  - This file consists of 5 search queries.
  - Each query point is close to one of the clusters in the point_5-4.txt file.
  - There is a one-to-one correspondence between the query points and the clusters.

- [ground-truth_5-4.txt](./ground-truth_5-4.txt):
  - This file contains ground truth data of the 5 queries.
  - The first half of the file lists the ground truth nearest neighbor IDs.
  - The second half of the file lists the ground truth distances.
  - For example, the first line is for the ground truth nearest neighbor IDs of the first query point. The sixth line contains the ground truth distances of the first query point.

- [all-distance-pairs_5-4.txt](./all-distance-pairs_5-4.txt):
  - This file contains all possible distance pairs between the input points in the dataset.