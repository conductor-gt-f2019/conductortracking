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
    hand_x = "left_x" if hand == "left" else "right_x"
    hand_y = "left_y" if hand == "left" else "right_y"

    # Remove points that are precisely at zero
    seg = segment[(segment.loc[:, hand_x] != 0).values * (segment.loc[:, hand_y] != 0).values].copy()

    # Normalize segment data into the range of [0, 500 - 100]
    y_min = seg.loc[:, hand_y].min()
    x_min = seg.loc[:, hand_x].min()
    seg.loc[:, hand_y] = seg[hand_y].apply(lambda v: v - y_min)
    seg.loc[:, hand_x] = seg[hand_x].apply(lambda v: v - x_min)

    y_max = seg.loc[:, hand_y].max()
    x_max = seg.loc[:, hand_x].max()
    seg.loc[:, hand_y] = seg[hand_y].apply(lambda v: v / y_max * (m_size[0] - 100 - 1))
    seg.loc[:, hand_x] = seg[hand_x].apply(lambda v: v / x_max * (m_size[1] - 100 - 1))

    # Cast to int so points can be used ast array indices
    int_segment = seg.astype('int32')

    # Create accumulator matrix
    acc = np.zeros(m_size)
    acc[int_segment.loc[:, hand_y] + 50, int_segment.loc[:, hand_x] + 50] = 1

    return acc

def save_image(acc: np.ndarray, filepath: str, reverse: False):
    three_channel = np.array([acc, acc, acc])
    three_channel = three_channel.transpose(1, 2, 0)
    three_channel = three_channel * 255
    if reverse == False:
        cv2.imwrite(filepath, three_channel)
    else:
        cv2.imwrite(filepath, cv2.flip(three_channel, 1)) # Save a horizontally flipped image too.

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
    train_accs = acc_matrices[:ind75]
    val_accs = acc_matrices[ind75:]

    for j in range(len(train_accs)):
        filepath = "data/train/" + timesignature + "/" + filename[:-4] + "_" + str(j) + ".png"
        r_filepath = "data/train/" + timesignature + "/" + filename[:-4] + "_r" + str(j) + ".png"
        save_image(train_accs[j], filepath, False)
        print("Saved: ", filepath)
        save_image(train_accs[j], r_filepath, True)
        print("Saved: ", r_filepath)

    for j in range(len(val_accs)):
        filepath = "data/val/" + timesignature + "/" + filename[:-4] + "_" + str(j + ind75) + ".png"
        r_filepath = "data/val/" + timesignature + "/" + filename[:-4] + "_r" + str(j + ind75) + ".png"
        save_image(val_accs[j], filepath, False)
        print("Saved: ", filepath)
        save_image(val_accs[j], r_filepath, True)
        print("Saved: ", r_filepath)

# %%
if __name__ == "__main__":    
    saveTrainAndVal("pose_data/data_both_hands_2-2_g.csv", "data_both_hands_2-2_g.csv", "right", "2_2")
    saveTrainAndVal("pose_data/data_both_hands_2-2_g.csv", "data_both_hands_2-2_g.csv", "left", "2_2")
    # saveTrainAndVal("pose_data/data_left_hand_3-4_g.csv", "data_left_hand_3-4_g.csv", "left", "3_4")
    # saveTrainAndVal("pose_data/data_right_hand_3-4_g.csv", "data_right_hand_3-4_g.csv", "right", "3_4")
    # saveTrainAndVal("pose_data/data_left_hand_p1_4-4_g.csv", "data_left_hand_p1_4-4_g.csv", "left", "4_4")
    # saveTrainAndVal("pose_data/data_left_hand_p2_4-4_g.csv", "data_left_hand_p2_4-4_g.csv", "left", "4_4")
    # saveTrainAndVal("pose_data/data_right_hand_4-4_g.csv", "data_right_hand_4-4_g.csv", "right", "4_4")
    saveTrainAndVal("pose_data/data_right_hand_pt_4-4.csv", "data_right_hand_pt_4-4.csv", "right", "4_4")
    saveTrainAndVal("pose_data/data_ragh_left_p1_3-4_g.csv", "data_ragh_left_p1_3-4_g.csv", "left", "3_4")
    saveTrainAndVal("pose_data/data_ragh_right_3-4_g.csv", "data_ragh_right_3-4_g.csv", "right", "3_4")
    saveTrainAndVal("pose_data/data_ragh_right_p2_3-4_g.csv", "data_ragh_right_p2_3-4_g.csv", "right", "3_4")
