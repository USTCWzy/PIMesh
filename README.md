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

![](./images/dataset.gif)

### Contents and Attributes

The data are organized in experimental groups and `.rar` format (each `.rar` file means a group in the experiment) . In each compressed file, there are:

![](./images/file_list.png)

* `depth`: synchronized collected depth images

* `rgb_image`:  synchronized collected RGB images

* `pressure_correct.npy`: synchronized collected pressure images, organized in (-1, 56, 40) `np.array` format

* `labels.npz`：annotated labels including participant information, postures, 2D keypoints, 3D shape

    * Reading codes

        ```python
        '''
        import numpy as np
        
        path = r'1/1/labels.npz'
        data = dict(np.load(path, allow_pickle=True))
        
        # data is a dictionary instance
        
        print(data.keys())
        # dict_keys(['gender', 'height', 'weight', 'label_pose', 'label_betas', 'label_trans', 'keypoints_pix', 'keypoints_pix_smooth', 'keypoints_meter', 'keypoints_meter_smooth', 'posture_index'])
        '''
        ```

    * keys

        * `gender`: gender of the current subject
        * `height`: height
        * `weight`: body weight
        * `label_pose`: SMPL pose parameter labels $\theta \in \mathbb{R}^{N \times 72}$ 
        * `label_betas`: SMPL shape parameter labels $\beta \in \mathbb{R}^{N \times 10}$
        * `label_trans`: root translation labels in the world coordinate system in SMPL $t \in \mathbb{R}^{N \times 3}$
        * `keypoints_pix`: 2D keypoint labels in the image coordinate system
        * `keypoints_pix_smooth`: 2D keypoint labels in the image coordinate system after Savitzky-Golay  filters (polyorder as 2
            and window size as 7)
        * `keypoints_meter`: 2D keypoint labels in the world coordinate (calculated by camera calibration and perspective transformation)
        * `keypoints_meter_smooth`: 2D keypoint labels in the world coordinate system after Savitzky-Golay  filters (polyorder as 2
            and window size as 7)

* `camera_params.npz`: camera calibration parameters and sensor positions in the world coordinate

    * Reading codes

        ```python
        '''
        import numpy as np
        
        path = r'1/1/camera_params.npz'
        data = dict(np.load(path, allow_pickle=True))
        
        # data is a dictionary instance
        
        print(data.keys())
        # dict_keys(['depth_cali', 'depth_bed', 'intParam', 'cali_board_image_position', 'cali_board_world_postion', 'cali_board_bed_corner_shift', 'sensor_number', 'sensor_interval_dis', 'cali_sensor_image_position', 'cali_sensor_position', 'infer_sensor_position', 'bed_left_corner_shift', 'bed_corner_shift'])
        '''
        ```

    * keys

* `valid_frame`



### Camera Calibration

TESTTEST

## Reorganized Datasets for PIMesh's Training, Validation, and Testing



## Contact

If you have any questions about the paper, code and dataset, please feel free to contact [wzy1999@mail.ustc.edu.cn](mailto:wzy1999@mail.ustc.edu.cn).





