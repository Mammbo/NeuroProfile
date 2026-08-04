# Daniel Alvarez
# Chunker.py
# split a given video into 100s chunks for the model to process with 10s of overlap at each step

import typing
import ffmpeg
import os
def chunk_video(video, window=100, overlap=10, out_dir='chunks'):
    # first the ingested video is passed to the function
    # get the total length of the video with ffmpeg
    probe = ffmpeg.probe(video)
    duration = float(probe['format']['duration'])

    # computer windows and divide that into an array of 100 second piece arrays so an array of array
        #ex: [[0, 100], [90, 190], [180, 280]]

    step = window - overlap
    windows = []
    start = 0
    while start < duration: 
        end = start + window
        windows.append([start, min(end, duration)])
        if end >= duration: 
            break
        start += step 
    
    # use ffmpeg to injust the mp4 file from video and cut the video into that many chunks that in the chunked seconds array above
    os.makedirs(out_dir, exist_ok=True)
    path_and_time = {}
    for i in range(len(windows)):
        out_path = os.path.join(out_dir, f'chunk_{i:03d}.mp4')
        (
            ffmpeg
            .input(video, ss=windows[i][0], t=windows[i][1] - windows[i][0])
            .output(out_path, c='copy')
            .overwrite_output()
            .run(quiet=True)
        )
        path_and_time[out_path] = windows[i][0]
    # return an output which is a list of paths and the absolute start_time, path is for the model, start time is for the stitcher
    return path_and_time

