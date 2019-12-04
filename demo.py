import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
import torchvision
import itertools
from main import data_transforms
from start_openpose import op, opWrapper
from get_pose_data import get_pose_data
from progress.bar import Bar
from collections import deque
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torchvision.models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)
model = model.to(device)
model.load_state_dict(torch.load('ct_model_adam_001lr.pt'))

class PointAccumulator:
    def __init__(self, maxSize, lookBackSize):
        self.maxSize = maxSize
        self.lookBackSize = lookBackSize
        self.xpoints = deque()
        self.ypoints = deque()
        self.x1stderiv = deque()
        self.y1stderiv = deque()
        self.x2ndderiv = deque()
        self.y2ndderiv = deque()
        self.allQueues = [self.xpoints, self.ypoints,
                    self.x1stderiv, self.y1stderiv,
                    self.x2ndderiv, self.y2ndderiv]
        
        self.maxWait = int(lookBackSize * 3)
        # Once an inflection point is detected, don't bother
        # checking again until this variable counts down to 0.
        self.wait = self.maxWait

        self.inflectionFrame = [0, 0]
    
    def popLeft(self):
        for q in self.allQueues: q.popleft()

    def addPoint(self, x, y):
        # If the queue is already full, get rid of an old point
        if len(self.xpoints) == self.maxSize:
            self.popLeft()
        
        # Add point
        self.xpoints.append(x)
        self.ypoints.append(y)

        # Compute first derivative
        if len(self.xpoints) > 1:
            self.x1stderiv.append(self.xpoints[-1] - self.xpoints[-2])
            self.y1stderiv.append(self.ypoints[-1] - self.ypoints[-2])
        
        # Compute second derivative
        if len(self.x1stderiv) > 1:
            self.x2ndderiv.append(self.x1stderiv[-1] - self.x1stderiv[-2])
            self.y2ndderiv.append(self.y1stderiv[-1] - self.y1stderiv[-2])
    
    def detectInflectionY(self):
        if self.wait > 0:
            self.wait -= 1
            self.inflectionFrame[0] -= 1
            self.inflectionFrame[1] -= 1

        # Can't compute inflection point without enough data
        if len(self.y2ndderiv) < self.lookBackSize:
            return False
        
        # Don't have to check inflection point if it was recently calculated
        if self.wait > 0:
            return False
        
        # Detect whether the y point is near the apex
        # In other words, in the top 1/3rd of the bounding box (assumption)
        np_y = np.array(self.ypoints)
        y_min = np.amin(np_y)
        y_max = np.amax(np_y)
        cutoff = (y_max - y_min) * 0.33 + y_min

        # End function early if not near apex
        if self.ypoints[-1] > cutoff:
            return False

        # Get sign of an old point
        oldSign = self.y2ndderiv[-self.lookBackSize] > 0
        # Get sign of the current point
        newSign = self.y2ndderiv[-1] > 0

        if oldSign != newSign:
            # Reset waiting time
            self.wait = self.maxWait
            self.inflectionFrame[0] = self.inflectionFrame[1]
            self.inflectionFrame[1] = 1
            return True
        else:
            return False

    def printAcc(self):
        for q in self.allQueues: print(q)

def getWristPoint(frame: np.ndarray, side: str):
    wristID = 4 if side == "right" else 7 if side == "left" else None
    pose_data = get_pose_data(frame)
    wrist_point = pose_data.poseKeypoints[0].reshape((-1, 3))[wristID]
    return wrist_point

def create_accumulator_matrix(xpoints, ypoints, m_size=(500, 500)) -> np.ndarray:

    # Normalize segment data into the range of [0, 500]
    x_min = np.amin(xpoints)
    y_min = np.amin(ypoints)
    xpoints = xpoints - x_min
    ypoints = ypoints - y_min

    x_max = np.amax(xpoints)
    y_max = np.amax(ypoints)
    xpoints = xpoints / x_max * (m_size[1] - 100 - 1)
    ypoints = ypoints / y_max * (m_size[0] - 100 - 1)
    xpoints = xpoints.astype(int)
    ypoints = ypoints.astype(int)

    # Create accumulator matrix
    acc = np.zeros((m_size[0], m_size[1], 3))
    acc[ypoints + 50, xpoints + 50, :] = 255

    return acc.astype(np.uint8)

def showInfo(frame, x, y, text):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (x, y)
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2

    cv2.putText(frame, text, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)


def readVideo(inputvideo: str, target_hand: str, outputvideo: str):
    cap = cv2.VideoCapture(inputvideo)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(outputvideo, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

    # Print error if video file couldn't be opened
    if (cap.isOpened() is False):
        print("Error opening video stream or file.")

    # Get an estimate of the number of frames in the video
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Create the point accumulator
    acc = PointAccumulator(128, 8)

    # Don't need to watch the whole thing
    count = 0

    with Bar('Processing:', max=total_frames) as bar:
        while(cap.isOpened()):
            ret, frame = cap.read()

            if ret is False:
                break

            if count > 1000:
                break

            # Get data for current frame
            left_hand = getWristPoint(frame, "left")
            right_hand = getWristPoint(frame, "right")

            # Show positions on cv2 frame
            cv2.circle(frame, (int(left_hand[0]), int(left_hand[1])), 5, (255, 0, 0), -1)
            cv2.circle(frame, (int(right_hand[0]), int(right_hand[1])), 5, (0, 255, 0), -1)

            hand = left_hand if target_hand == "left" else right_hand
            acc.addPoint(hand[0], hand[1])

            if len(acc.xpoints) > 100:
                xpoints = np.array(acc.xpoints)
                ypoints = np.array(acc.ypoints)

                # Remove x,y points where both are at zero. These are errors.
                bothNot0 = (xpoints != 0) * (ypoints != 0)
                xpoints = xpoints[bothNot0]
                ypoints = ypoints[bothNot0]

                for i in range(len(xpoints)):
                    cv2.circle(frame, (xpoints[i], ypoints[i]), 2, (255, 100, 255), -1)

                xmin = np.amin(xpoints)
                xmax = np.amax(xpoints)
                ymin = np.amin(ypoints)
                ymax = np.amax(ypoints)

                # Draw bounding box
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                

                pose_acc = create_accumulator_matrix(xpoints, ypoints)
                pose_im = Image.fromarray(pose_acc)
                tensor_im = data_transforms['val'](pose_im)
                image = tensor_im.to(device).unsqueeze(0)
                output = model(image)
                _, pred = torch.max(output, 1)
                predInd = pred.item()
                predLabel = ["2_2", "3_4", "4_4"][predInd]
                showInfo(frame, 10, frame.shape[0] - 30, "Predicted label: " + predLabel)
                print(["2_2", "3_4", "4_4"][predInd])

            out.write(frame)
            cv2.imshow("Pose Data", frame)
            k = cv2.waitKey(1)
            if k & 0xFF == ord('q'):
                break

            # Increase count variable
            bar.next()
            count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("inputvideo", type=str, help="Input video")
    parser.add_argument("hand", type=str, help="hand")
    parser.add_argument("outputvideo", type=str, help="Output video")
    args = parser.parse_args()
    inputvideo = args.inputvideo
    outputvideo = args.outputvideo
    hand = args.hand

    # Read and scrape data from video
    readVideo(inputvideo, hand, outputvideo)