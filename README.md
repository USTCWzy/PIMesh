# PIMesh: Video-based In-bed Human Shape Estimation from Pressure Images
Repo for Ubicomp2024 paper: "Seeing through the Tactile: 3D Human Shape Estimation from Temporal In-Bed Pressure Images" 

![](./images/pipeline.png)

## Temporal multi-modality In-bed Dataset (TIP)

![](./images/dataset_overview.png)

**Temporal human In-bed Pose dataset (TIP)**: a multi-modality dataset of 152K images from 9 subjects and 40 groups with diverse human representation ground truths (posture, joints, and 3D mesh). The download link is in [Seafile](http://210.45.71.78:8888/d/b16e65834409491a970f/).

### Features

* **Three Modalities**:  Pressure, RGB, and depth images.
* **Labels**: including 2D keypoint positions in image and world coordinate systems, 3D human shape labels (in SMPL), posture class, and body attributes.

    * 2D keypoint labels based on COCO17

    * 3D shape labels based on SMPL

    * posture categories including 28 static postures and 2 motions.

        ![](./images/postures.png)

        <center>28 static postures (17 supine postures, 7 side postures, and 5 prones) and 2 motions (crunches and push-ups)</center>

    * body attributes including **body weights** and **heights**

        |  Age  | Gender |        | Height (cm) |      |               | Weight (kg) |       |              |
        | :---: | :----: | :----: | :---------: | :--: | :-----------: | :---------: | :---: | :----------: |
        | Range |  Male  | Female |     Max     | Min  |  Mean (std)   |     Max     |  Min  |  Mean (std)  |
        | 22-24 |   4    |   5    |     179     | 152  | 166.17 (7.39) |    75.20    | 38.00 | 55.52 (9,59) |
* **Large-scale**: **152K** three-modality synchronized temporal images from **9** volunteers performing **30** postures in a total of **40** groups.

### Visualization

![](./images/a slice of our dataset.gif)

### Contents and Attributes

The data are organized in the group setting and `.rar` format. In each compressed file, there are:

![](./images/file_list.png)

* `depth`: synchronized collected depth images
* `rgb_image`:  synchronized collected RGB images
* `pressure_correct.npy`: synchronized collected pressure images, organized in (-1, 56, 40) `np.array` format
* 



### Camera Calibration

TESTTEST

## Reorganized Datasets for PIMesh's Training, Validation, and Testing



## Contact

If you have any questions about the paper, code and dataset, please feel free to contact [wzy1999@mail.ustc.edu.cn](mailto:wzy1999@mail.ustc.edu.cn).





