# AR Semantic Understanding

Semantic understanding for augmented reality (AR) applications.

This repository contains code for a project which builds a proof-of-concept pipeline to bring semantic understanding to AR. the `model` directory contains code to train a semantic segmentation model on indoor scene data. The `server` directory contains code to deploy an edge server to capture RGB-D frames from an Intel RealSense camera and transmit the result. The `magic_leap` folder contains a Unity application for the Magic Leap One to overlay the user's field of view with semantic information (i.e. a semantically segmented point cloud). The `mesh` directory contains some experimental code for meshing.

The dataset used and a pre-trained model are available at [https://github.com/jdechicchis/ar-semantic-understanding-data](https://github.com/jdechicchis/ar-semantic-understanding-data).
