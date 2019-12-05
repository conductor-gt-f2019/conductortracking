import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from start_openpose import op, opWrapper
from get_pose_data import get_pose_data
from progress.bar import Bar

def getWristPoint(frame: np.ndarray, side: str):
    wristID = 4 if side == "right" else 7 if side == "left" else None
    pose_data = get_pose_data(frame)
    wrist_point = pose_data.poseKeypoints[0].reshape((-1, 3))[wristID]
    return wrist_point

def readVideo(inputvideo: str, outputfolder: str):
    cap = cv2.VideoCapture("videos/" + inputvideo)

    # Print error if video file couldn't be opened
    if (cap.isOpened() is False):
        print("Error opening video stream or file.")

    # Create stack for holding the openpose data
    frame_data = pd.DataFrame()

    # Get an estimate of the number of frames in the video
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    count = 0
    
    with Bar('Processing:', max=total_frames) as bar:
        while(cap.isOpened()):
            ret, frame = cap.read()

            if ret is False:
                break

            # Get data for current frame
            left_hand = getWristPoint(frame, "left")
            right_hand = getWristPoint(frame, "right")

            # Add positions to frame_data
            frame_data.loc[count, "left_x"] = left_hand[0]
            frame_data.loc[count, "left_y"] = left_hand[1]
            frame_data.loc[count, "right_x"] = right_hand[0]
            frame_data.loc[count, "right_y"] = right_hand[1]
            # frame_data.loc[count, "segment"] = 0

            # # Show positions on cv2 frame
            # cv2.circle(frame, (int(left_hand[0]), int(left_hand[1])), 5, (255, 0, 0), -1)
            # cv2.circle(frame, (int(right_hand[0]), int(right_hand[1])), 5, (0, 255, 0), -1)
            # cv2.imshow("Pose Data", frame)

            # k = cv2.waitKey(1)
            # if k & 0xFF == 32:
            #     frame_data.loc[count, "segment"] = 1
            # elif k & 0xFF == ord('q'):
            #     break

            # Increase count variable
            count += 1
            bar.next()

    # Save csv
    frame_data.to_csv(outputfolder + "/op_" + inputvideo[:-4] + ".csv", index=False)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("inputvideo", type=str, help="Input video")
    parser.add_argument("outputfolder", type=str, help="Output folder")
    args = parser.parse_args()
    inputvideo = args.inputvideo
    outputfolder = args.outputfolder

    # Read and scrape data from video
    readVideo(inputvideo, outputfolder)