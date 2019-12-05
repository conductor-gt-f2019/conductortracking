# Conductor Gesture Tracking CV Project
The website for this project can be found at: https://conductor-gt-f2019.github.io/conductortracking/

## Data Collection Process
For each time signature, we recorded approximately 15 minutes of footage (480p, 30fps) of ourselves performing the corresponding gesture.

We then used OpenPose (https://github.com/CMU-Perceptual-Computing-Lab/openpose) to extract the x and y coordinates of our wrists in each video. The script we used for coordinate extraction can be found in [get_pose_data_from_video.py](get_pose_data_from_video.py).

Using [verify_segment_time.py](verify_segment_time.py), we annotated the start of each gesture loop. We then processed the annotated data using [accumulator.py](accumulator.py) to produce accumulator matrices containing a mapping of the gesture's points. The accumulator matrices were saved as images to the data folder for use in the training and validation datasets. 75% of each gesture were used for training, and the remaining 25% were used for validation.

| Gesture | # training | # validation | total |
| ------- | ---------- | -------------| ----- |
| 2/2     | 354        | 119          | 473   |
| 3/4     | 296        | 100          | 396   |
| 4/4     | 250        | 84           | 334   |


Link to data: https://drive.google.com/drive/folders/15A89ZKPJKIkJ8ThkmoGz7r9pBUl_xM1a?usp=sharing 


## Model
https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
