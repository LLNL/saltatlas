## 5-4 Dataset Overview

The 5-4 dataset is a collection of data files related to a set of **5** communities, each containing **4** points.
The dataset is used for search queries and consists of the following files:

- [point_5-4.txt](./point_5-4.txt):
  - This file contains the feature vectors (5 dimensions) for the dataset. 
  - The file is in white-space separated value (WSV) format.
  - The distance metric is L2 (Euclidean) distance.

- [query_5-4.txt](./query_5-4.txt):
  - This file consists of 5 search queries.
  - Each query point is close to one of the communities in the point_5-4.txt file.
  - There is a one-to-one correspondence between the query points and the communities.

- [ground-truth_5-4.txt](./ground-truth_5-4.txt):
  - This file contains ground truth data.
  - The first half of the file lists the ground truth nearest neighbor IDs.
  - The second half of the file lists the ground truth distances.
  - Each line corresponds to each query.

- [all-distance-pairs_5-4.txt](./all-distance-pairs_5-4.txt):
  - This file contains all possible distance pairs between the input points in the dataset.