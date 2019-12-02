# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import cv2
import random
from typing import List
from scipy import signal

def segment_data(filepath: str) -> List[pd.DataFrame]:
    '''Segment the full video's pose data by the start of each gesture loop.'''
    df = pd.read_csv(filepath)

    segInds = df.loc[df['segment'] == 1.0, :].index
    segments = []

    for i in range(len(segInds) - 1):
        df_seg = df.loc[segInds[i]:segInds[i+1], :]
        segments.append(df_seg)

    return segments

def create_accumulator_matrix(segment: pd.DataFrame, hand: str, m_size=(500, 500)) -> np.ndarray:
    hand_x = 0 if hand == "left" else 2
    hand_y = 1 if hand == "right" else 3

    # Remove points that are precisely at zero
    segment = segment.loc[(segment.iloc[:, hand_x] != 0).values * (segment.iloc[:, hand_y] != 0).values, :]

    # Normalize segment data into the range of [0, 500]
    y_min = segment.iloc[:, hand_y].min()
    x_min = segment.iloc[:, hand_x].min()
    segment.iloc[:, hand_y] = segment.iloc[:, hand_y] - y_min
    segment.iloc[:, hand_x] = segment.iloc[:, hand_x] - x_min

    y_max = segment.iloc[:, hand_y].max()
    x_max = segment.iloc[:, hand_x].max()
    segment.iloc[:, hand_y] = segment.iloc[:, hand_y] / y_max * (m_size[0] - 1)
    segment.iloc[:, hand_x] = segment.iloc[:, hand_x] / x_max * (m_size[1] - 1)

    # Cast to int so points can be used ast array indices
    int_segment = segment.astype('int32')

    # Create accumulator matrix
    acc = np.zeros(m_size)
    acc[int_segment.iloc[:, hand_y], int_segment.iloc[:, hand_x]] = 1

    return acc

def save_image(acc: np.ndarray, filepath: str):
    cv2.imwrite(filepath, acc * 255)

def saveTrainAndVal(filepath:str, filename:str, hand:str, timesignature: str):
    # filepath = "pose_data/data_both_hands_2-2_g.csv"
    # filename = "data_both_hands_2-2_g.csv"
    segments = segment_data(filepath)
    acc_matrices = []
    for segment in segments:
        acc = create_accumulator_matrix(segment, hand)
        acc_matrices.append(acc)

    # Shuffle the data
    random.shuffle(acc_matrices)

    # Prepare 75% of data for data folder
    # Prepare the remaining 25% for val folder
    ind75 = int(len(acc_matrices) * 0.75)
    print(filename, len(acc_matrices), ind75, len(acc_matrices) - ind75)
    train_accs = acc_matrices[:ind75]
    val_accs = acc_matrices[ind75:]

    for j in range(len(train_accs)):
        filepath = "data/train/" + timesignature + "/" + filename[:-4] + "_" + str(j) + ".png"
        save_image(train_accs[j], filepath)
        print("Saved: ", filepath)

    for j in range(len(val_accs)):
        filepath = "data/val/" + timesignature + "/" + filename[:-4] + "_" + str(j + ind75) + ".png"
        save_image(val_accs[j], filepath)
        print("Saved: ", filepath)

# %%
if __name__ == "__main__":    
    saveTrainAndVal("pose_data/data_both_hands_2-2_g.csv", "data_both_hands_2-2_g.csv", "right", "2_2")
    saveTrainAndVal("pose_data/data_both_hands_2-2_g.csv", "data_both_hands_2-2_g.csv", "left", "2_2")
    saveTrainAndVal("pose_data/data_left_hand_3-4_g.csv", "data_left_hand_3-4_g.csv", "left", "3_4")
    saveTrainAndVal("pose_data/data_right_hand_3-4_g.csv", "data_right_hand_3-4_g.csv", "right", "3_4")
    saveTrainAndVal("pose_data/data_left_hand_p1_4-4_g.csv", "data_left_hand_p1_4-4_g.csv", "left", "4_4")
    saveTrainAndVal("pose_data/data_left_hand_p2_4-4_g.csv", "data_left_hand_p2_4-4_g.csv", "left", "4_4")
    saveTrainAndVal("pose_data/data_right_hand_4-4_g.csv", "data_right_hand_4-4_g.csv", "right", "4_4")