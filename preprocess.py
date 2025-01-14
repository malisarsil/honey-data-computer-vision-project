import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from helpers import *
import pandas as pd
import os
from joblib import Parallel, delayed



folderNames = ["Room Temparature", "Boiled Water"]


data_path = '/content/drive/MyDrive/Honey/data/water surface method'
partitionPath = '/content/drive/MyDrive/Honey/data/partition'
corner_file_path = '/content/drive/MyDrive/Colab Notebooks/cornerInfo.xlsx'
honey_drop_info_path = '/content/drive/MyDrive/Colab Notebooks/dropInfo.xlsx'
ground_truth_file_path = '/content/drive/MyDrive/Colab Notebooks/Bal_AI.xlsx'


video_files = os.listdir(partitionPath)  # Replace this with os.listdir(path) for actual files.
conditionsFolder = ['Boiled Water']#os.listdir(data_path)
corner_df = pd.read_excel(corner_file_path)
honey_drop_info_df = pd.read_excel(honey_drop_info_path)
ground_truth_file_df = pd.read_excel(ground_truth_file_path)
specialCorners = corner_df['File Name']


video_length = 9 #number of frames

k_values = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]# Frame indexes to extract in each second

# Function to process a single video file  
def process_video_file(data_path, conditionFolder, sampleName, sampleFileName, partitionPath, ground_truth_file_df, honey_drop_info_df, corner_df, specialCorners, k_values, video_length):
    conditionPath = os.path.join(data_path, conditionFolder)
    sampleFolderPath = os.path.join(conditionPath, sampleName)
    videos_saved = os.listdir(partitionPath)
    frame_lists = {k: [] for k in k_values}

    contains_string = any(sampleFileName.split('.')[0] in item for item in videos_saved)

    if not contains_string:
        valueCatch = sampleName
        matched_rows = ground_truth_file_df.loc[
            (ground_truth_file_df['Numune Kodu 1'] == valueCatch) | (ground_truth_file_df['Numune Kodu 2'] == int(valueCatch))
        ]
        target_label = matched_rows['Target Label'].values[0]
        videoPath = os.path.join(sampleFolderPath, sampleFileName)
        print('Partitioning for:', sampleName, sampleFileName)

        cap = cv2.VideoCapture(videoPath)

        getStartingFrame_Row = honey_drop_info_df[honey_drop_info_df["video_path"].str.contains(sampleFileName, case=False, na=False)]
        if len(getStartingFrame_Row) == 1:
            matching_row_drop_instant_Frame_num = int(getStartingFrame_Row['frame instant of honey drop'])
            fps = int(getStartingFrame_Row['fps'])
        else:
            print('No drop info for:', sampleFileName)
            return

        startFrame = 0
        frame_count = 0
        frame_count_stack = 0
        stacked_cropped_frames = []
        video_length_seconds_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if startFrame == 0:
                startFrame = 1
                height = frame.shape[0]
                width = frame.shape[1]
                frame, height, width = checkInputImage(frame, height, width)

                matching_indices = [index for index, element in enumerate(specialCorners) if sampleFileName in element]
                if matching_indices:
                    result_row = corner_df.iloc[matching_indices[0], :]
                    coordinates = [
                        [int(result_row['cord0'].split(',')[0]), int(result_row['cord0'].split(',')[1])],
                        [int(result_row['cord1'].split(',')[0]), int(result_row['cord1'].split(',')[1])],
                        [int(result_row['cord2'].split(',')[0]), int(result_row['cord2'].split(',')[1])],
                        [int(result_row['cord3'].split(',')[0]), int(result_row['cord3'].split(',')[1])],
                    ]
                else:
                    bottommost_row_index, uppermost_row_index = getUpperandLowerRowBoundary(
                        frame, height, width, threshold=150, plotImage=False
                    )
                    coordinates = find_side_lines(frame, 'placeholder', bottommost_row_index, uppermost_row_index, threshold=70, min_line_length=50, max_line_gap=10, plotEdgeDetectionResult=False)

            if frame_count > matching_row_drop_instant_Frame_num and video_length_seconds_count < video_length:
                frame, height, width = checkInputImage(frame, height, width)

                imageName = f'{sampleName}-{sampleFileName.split(".")[0]}.png'
                masked_image = saveSegmentedImage(frame, height, width, coordinates, imageName)
                cropped_image = remove_bottom_black(masked_image)
                stacked_cropped_frames.append(cropped_image)
                frame_count_stack += 1

                if frame_count_stack % fps == 0:
                    for k in k_values:
                        if k < len(stacked_cropped_frames):
                            frame_lists[k].append(stacked_cropped_frames[k])
                    video_length_seconds_count += 1
                    stacked_cropped_frames = []

            if video_length_seconds_count == video_length - 1:
                output_dir = partitionPath
                for k, frames in frame_lists.items():
                    if frames:
                        height, width, channels = frames[0].shape
                        output_file = os.path.join(output_dir, f'{target_label}-{conditionFolder}-{sampleName}-{sampleFileName.split(".")[0]}-{k}.mp4')
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
                        for frame in frames:
                            video_writer.write(frame)
                        video_writer.release()
                        print(f"Video saved: {output_file}")
                break

            frame_count += 1

        cap.release()
    else:
        print(f"{sampleFileName} is already processed.")

# Wrapper function to call process_video_file for all videos
def process_all_videos(data_path, partitionPath, conditionsFolder, ground_truth_file_df, honey_drop_info_df, corner_df, specialCorners, k_values, video_length):
    Parallel(n_jobs=-1)(
        delayed(process_video_file)(
            data_path, conditionFolder, sampleName, sampleFileName, partitionPath, ground_truth_file_df,
            honey_drop_info_df, corner_df, specialCorners, k_values, video_length
        )
        for conditionFolder in conditionsFolder
        for sampleName in os.listdir(os.path.join(data_path, conditionFolder))
        for sampleFileName in os.listdir(os.path.join(data_path, conditionFolder, sampleName))
    )

# Usage example
process_all_videos(data_path, partitionPath, conditionsFolder, ground_truth_file_df, honey_drop_info_df, corner_df, specialCorners, k_values, video_length)
