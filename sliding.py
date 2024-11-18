#%%
import os
import imageio
import glob
import cv2
import numpy as np
from PIL import Image

def video_to_images(video_path):
    """Parses a video into individual image frames."""

    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0

    images = [image]

    while success:
        success, image = vidcap.read()
        count += 1
        if image is not None:
            images.append(image)

    vidcap.release()

    return images

def make_mp4(frames, savepath, fps=30):
    frames = [Image.fromarray(frame) for frame in frames]
    frame_one = frames[0]
    videodims = tuple(np.array(frame_one).shape[:-1][::-1])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')    
    video = cv2.VideoWriter(f"{savepath}",fourcc, fps,videodims)
    for frame in frames:
        # draw frame specific stuff here.
        video.write(np.array(frame))
    video.release()

splat_video = 'renders/splat/render.mp4'
ellipsoid_color_video = 'renders/mesh_trimesh/render.mp4'
ellipsoid_video = 'renders/mesh_simple/render.mp4'

splat_images = video_to_images(splat_video)[:400]
ellipsoid_color_images = video_to_images(ellipsoid_color_video)[:400]
ellipsoid_images = video_to_images(ellipsoid_video)[:400]

# Overlay images on top of each other in sweeping fashion
final_frames = []
H, W = splat_images[0].shape[:2]
len_frames = len(splat_images)

buffer = [150, 250, 350]
for i, (frame1, frame2, frame3) in enumerate(zip(splat_images, ellipsoid_images, ellipsoid_color_images)):
    final_frame = np.zeros_like(frame1)

    frame2 = cv2.resize(frame2, (W, H))
    frame3 = cv2.resize(frame3, (W, H))

    # compute progress
    if i < buffer[0]:
        final_frame = frame1

    if i < buffer[1] and i >= buffer[0]:
        cutoff = int(W*(i - buffer[0])/(buffer[1] - buffer[0]))

        # Sweep goes from left to right
        final_frame[:, :cutoff] = frame2[:, :cutoff]
        final_frame[:, cutoff:] = frame1[:, cutoff:]

        left_bar = np.clip(cutoff-1, 0, W-1)
        right_bar = np.clip(cutoff+1, 0, W-1)
        final_frame[:, left_bar:right_bar] = 255//4

    elif i >= buffer[1]:
        cutoff = int(W*(i-buffer[1])/(buffer[2] - buffer[1]))

        # Sweep goes from right to left
        final_frame[:, cutoff:] = frame2[:, cutoff:]
        final_frame[:, :cutoff] = frame3[:, :cutoff]
    
        left_bar = np.clip(cutoff-1, 0, W-1)
        right_bar = np.clip(cutoff+1, 0, W-1)
        final_frame[:, left_bar:right_bar] = 255//4

    final_frames.append(final_frame)

make_mp4(final_frames, 'renders/sweep.mp4', fps=24)

# %%

video1 = 'renders/drone_move/render.mp4'
video2 = 'renders/traj_viz/render.mp4'

frames1 = video_to_images(video1)[-300:]
frames2 = video_to_images(video2)[-300:]

# Overlay images on top of each other in sweeping fashion
final_frames = []
H, W = frames1[0].shape[:2]
len_frames = len(frames1)

left_buffer = 150
right_buffer = 75
for i, (frame1, frame2) in enumerate(zip(frames1, frames2)):
    final_frame = np.zeros_like(frame1)

    # compute progress
    if i > left_buffer and i < len_frames - right_buffer:
        cutoff = int(W*(i - left_buffer)/(len_frames -  (left_buffer + right_buffer)))

        # Sweep goes from left to right
        final_frame[:, :cutoff] = frame2[:, :cutoff]
        final_frame[:, cutoff:] = frame1[:, cutoff:]

        left_bar = np.clip(cutoff-1, 0, W-1)
        right_bar = np.clip(cutoff+1, 0, W-1)
        final_frame[:, left_bar:right_bar] = 255//4

    elif i <= left_buffer:
        final_frame = frame1
    else:
        final_frame = frame2

    final_frames.append(final_frame)

make_mp4(final_frames, 'renders/sweep2.mp4', fps=30)
#%%