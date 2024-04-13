# PIMesh: Video-based In-bed Human Shape Estimation from Pressure Images
Repo for Ubicomp2024 paper: "Seeing through the Tactile: 3D Human Shape Estimation from Temporal In-Bed Pressure Images" 



## Temporal multi-modality In-bed Dataset (TIP)

![](images\dataset_overview.png)

### Features

* **Three Modalities**:  Pressure, RGB, and depth images.
* **Labels**: including 2D keypoint positions in image and world coordinate systems, 3D human shape labels (in SMPL), posture class, and body attributes.
    * 2D keypoint labels based on COCO17
    * 3D shape labels based on SMPL
    * posture categories including 28 static postures and 2 motions.
    * body attributes including **body weights** and **heights**
* **Large-scale**: **152K** three-modality synchronized temporal images from **9** volunteers performing **30** postures in a total of **40** groups.

