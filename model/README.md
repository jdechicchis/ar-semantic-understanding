# Model and Data

## Model Classes

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

9109 total 224(w) x 224(h) JPEG images. Class name to ID mapping are in `classes.json`. The original data contained 10335 images but images with no annotations (1226) were discarded.

Pixel-level class instances:

| Class              | ID | # of Instances | % of Instances |
|--------------------|----|----------------|----------------|
| Unknown            | 0  | 336035069      | 73.52%         |
| Bookshelf          | 1  | 2273833        | 0.50%          |
| Desk/Table/Counter | 2  | 47156248       | 10.32%         |
| Chair              | 3  | 44957965       | 9.84%          |
| Book/Paper         | 4  | 2891720        | 0.63%          |
| Picture            | 5  | 2838463        | 0.62%          |
| Window             | 6  | 11808008       | 2.58%          |
| Door               | 7  | 9091878        | 1.99%          |

Data mean and standard deviation:

|                    | Red      | Green    | Blue     |
|--------------------|----------|----------|----------|
| Mean               | 0.491024 | 0.455375 | 0.427466 |
| Standard Deviation | 0.262995 | 0.267877 | 0.270293 |

## Data Directory

The data directory is set up as follows:

```text
data -
     |---- classes.json
     |---- invalid_images.json
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

## Annotation Format

Annotation files contains a dictionary with the following keys and values:

- `id`: ID of annotation (string)
- `image_name`: Name of image associated with the annotation (string)
- `annotation`: A 224(h) x 224(w) array indexed by [h][w] where each entry is the class ID of the corresponding pixel.

Training class weights (based off overall pixel-level class instances):

0: 336035069/(9109*224*224) = 0.73522093437     => 0.10576790436/0.73522093437 = 0.14385866807
1: 2273833/(388*224*224) = 0.11679676199        => 0.10576790436/0.11679676199 = 0.90557223126
2: 47156248/(6403*224*224) = 0.14677757377      => 0.10576790436/0.14677757377 = 0.72059989577
3: 44957965/(5675*224*224) = 0.15788640724      => 0.10576790436/0.15788640724 = 0.66989873421
4: 2891720/(1924*224*224) = 0.0299540213        => 0.10576790436/0.0299540213  = 3.53100851804
5: 2838463/(1357*224*224) = 0.04168764416       => 0.10576790436/0.04168764416 = 2.53715235032
6: 11808008/(2484*224*224) = 0.09473904673      => 0.10576790436/0.09473904673 = 1.11641301038
7: 9091878/(2032*224*224)= 0.08917309888        => 0.10576790436/0.08917309888 = 1.18609654356

median = 0.10576790436

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

## Train/Test Split

Data was randomly split into train (8198 images) and test (911 images) set (90/10 split). The split is in `train_test_data_split.json`.

Train set pixel-level class instances:

| Class              | ID | # of Instances | % of Instances |
|--------------------|----|----------------|----------------|
| Unknown            | 0  | 302072880      | 73.44%         |
| Bookshelf          | 1  | 2091497        | 0.51%          |
| Desk/Table/Counter | 2  | 42791040       | 10.40%         |
| Chair              | 3  | 40423443       | 9.83%          |
| Book/Paper         | 4  | 2696598        | 0.66%          |
| Picture            | 5  | 2538673        | 0.62%          |
| Window             | 6  | 10658005       | 2.59%          |
| Door               | 7  | 8070712        | 1.96%          |

Test set pixel-level class instances:

| Class              | ID | # of Instances | % of Instances |
|--------------------|----|----------------|----------------|
| Unknown            | 0  | 33962189       | 73.97%         |
| Bookshelf          | 1  | 182336         | 0.40%          |
| Desk/Table/Counter | 2  | 4365208        | 9.51%          |
| Chair              | 3  | 4534522        | 9.88%          |
| Book/Paper         | 4  | 195122         | 0.42%          |
| Picture            | 5  | 299790         | 0.65%          |
| Window             | 6  | 1150003        | 2.50%          |
| Door               | 7  | 1021166        | 2.22%          |
