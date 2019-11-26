# AR Semantic Understanding

Semantic understanding for augmented reality (AR) applications.

## Model and Data

### Model Classes

| Class              | ID | Approx # of Instances |
|--------------------|----|-----------------------|
| Unknown            | 0  | N/A                   |
| Bookshelf          | 1  | 721                   |
| Desk/Table/Counter | 2  | 15467                 |
| Chair              | 3  | 27822                 |
| Book/Paper         | 4  | 6068                  |
| Picture            | 5  | 3908                  |
| Window             | 6  | 5506                  |
| Door               | 7  | 3849                  |

10335 total 224(w) x 224(h) JPEG images. Class name to ID mapping are in `classes.json`.

Pixel-level class instances:

| Class              | ID | # of Instances | % of Instances |
|--------------------|----|----------------|----------------|
| Unknown            | 0  | 397550845      | 76.66%         |
| Bookshelf          | 1  | 2273833        | 0.44%          |
| Desk/Table/Counter | 2  | 47156248       | 9.09%          |
| Chair              | 3  | 44957965       | 8.67%          |
| Book/Paper         | 4  | 2891720        | 0.56%          |
| Picture            | 5  | 2838463        | 0.55%          |
| Window             | 6  | 11808008       | 2.28%          |
| Door               | 7  | 9091878        | 1.75%          |

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
- `annotation`: A 224(w) x 224(h) array indexed by [h][w] where each entry is the class ID of the corresponding pixel.

Training class weights (based off overall pixel-level class instances):

| Class              | ID | Weight |
|--------------------|----|--------|
| Unknown            | 0  | 1      |
| Bookshelf          | 1  | 175    |
| Desk/Table/Counter | 2  | 8      |
| Chair              | 3  | 9      |
| Book/Paper         | 4  | 137    |
| Picture            | 5  | 140    |
| Window             | 6  | 34     |
| Door               | 7  | 44     |

### Train/Test Split

Data was randomly split into train (9300 images) and test (1035 images) set (90/10 split). The split is in `train_test_data_split.json`.

Train set pixel-level class instances:

| Class              | ID | # of Instances | % of Instances |
|--------------------|----|----------------|----------------|
| Unknown            | 0  | 357865897      | 76.69%         |
| Bookshelf          | 1  | 1932458        | 0.41%          |
| Desk/Table/Counter | 2  | 42494443       | 9.11%          |
| Chair              | 3  | 40422494       | 8.66%          |
| Book/Paper         | 4  | 2596679        | 0.56%          |
| Picture            | 5  | 2526685        | 0.54%          |
| Window             | 6  | 10618275       | 2.28%          |
| Door               | 7  | 8179869        | 1.75%          |

Test set pixel-level class instances:

| Class              | ID | # of Instances | % of Instances |
|--------------------|----|----------------|----------------|
| Unknown            | 0  | 39684948       | 76.42%         |
| Bookshelf          | 1  | 341375         | 0.66%          |
| Desk/Table/Counter | 2  | 4661805        | 8.98%          |
| Chair              | 3  | 4535471        | 8.73%          |
| Book/Paper         | 4  | 295041         | 0.57%          |
| Picture            | 5  | 311778         | 0.60%          |
| Window             | 6  | 1189733        | 2.29%          |
| Door               | 7  | 912009         | 1.76%          |
