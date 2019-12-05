import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from progress.bar import Bar

def verifyVideoAndCSV(inputvideo: str, inputcsv:str):
    df = pd.read_csv(inputcsv)
    cap = cv2.VideoCapture(inputvideo)

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

            # Draw points onto cv2 frame
            if df.loc[count, "segment"] == 1:
                cv2.circle(frame, (int(df.loc[count, "left_x"]), int(df.loc[count, "left_y"])), 5, (0, 0, 255), -1)
                cv2.circle(frame, (int(df.loc[count, "right_x"]), int(df.loc[count, "right_y"])), 5, (0, 0, 255), -1)
            else:
                cv2.circle(frame, (int(df.loc[count, "left_x"]), int(df.loc[count, "left_y"])), 5, (255, 0, 0), -1)
                cv2.circle(frame, (int(df.loc[count, "right_x"]), int(df.loc[count, "right_y"])), 5, (0, 255, 0), -1)
            cv2.imshow("Verification", frame)

            # Set up segment
            k = cv2.waitKey(28)
            if k & 0xFF == ord('q'):
                break

            # Increase count variable
            count += 1
            bar.next()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("inputvideo", type=str, help="Input video")
    parser.add_argument("inputcsv", type=str, help="Input csv")
    args = parser.parse_args()
    inputvideo = args.inputvideo
    inputcsv = args.inputcsv

    # Read and scrape data from video
    verifyVideoAndCSV(inputvideo, inputcsv)