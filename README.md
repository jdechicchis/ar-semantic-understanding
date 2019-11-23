# AR Semantic Understanding

Semantic understanding for augmented reality (AR) applications.

## Model and Data

### Model Classes

| Class              | ID | Approx Num Instances |
|--------------------|----|----------------------|
| Unknown            | 0  | N/A                  |
| Bookshelf          | 1  | 721                  |
| Desk/Table/Counter | 2  | 15467                |
| Chair              | 3  | 27822                |
| Book/Paper         | 4  | 6068                 |
| Picture            | 5  | 3908                 |
| Window             | 6  | 5506                 |
| Door               | 7  | 3849                 |

10335 total 800(w) x 500(h) JPEG images. Class name to ID mapping are in `classes.json`.

### Data Directory

The data directory is set up as follows:

```text
data -
     |---- classes.json
     |---- train_test_data_split.json
     |---- annotations
           |---- 00000.json
           |---- 00001.json
           |---- 00002.json
           |---- ...
           |---- 10335.json
     |---- images
           |---- 00000.jpg
           |---- 00001.jpg
           |---- 00002.jpg
           |---- ...
           |---- 10335.jpg
```

### Annotation Format

Annotation files contains a dictionary with the following keys and values:

- `id`: ID of annotation (string)
- `image_name`: Name of image associated with the annotation (string)
- `annotation`: A 800(w) x 500(h) array indexed by [w][h] where each entry is the class ID of the corresponding pixel.

### Train/Test Split

Data was randomly split into train (9300 images) and test (1035 images) set (90/10 split). The split is in `train_test_data_split.json`.
