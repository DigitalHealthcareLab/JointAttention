#wget   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 # DOWNLOAD LINK
#bunzip2 /content/shape_predictor_68_face_landmarks.dat.bz2

datFile =  "/content/shape_predictor_68_face_landmarks.dat"

from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import dlib

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import PIL
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
import shutil

root_path = Path('.')
for target_name in ['label', 'sev_ados'] :
    for target_task in ['IJA', 'RJA_LOW', 'RJA_HIGH'] : 
          for target_fold in range(10) : 
              if target_task == "IJA" : 
                  target_patient_ids = ["B702", "C129"]
              elif target_task == "RJA_LOW" : 
                  target_patient_ids = ['B704', 'C129']
              elif target_task == 'RJA_HIGH' : 
                  target_patient_ids = ['C129', 'D710']

              for target_patient_id in target_patient_ids : 
                  print(target_name, target_task, target_fold, target_patient_id)
                  target_fold_name = f"fold_{target_fold}"

                  target_path = root_path.joinpath(target_name, target_task, target_fold_name, target_patient_id)
                  gradcam_video_paths = sorted(np.array([path for path in target_path.glob('*.npy') if "raw" not in path.stem]))
                  raw_video_paths = sorted( np.array(list(target_path.glob('*raw.npy'))))
                  gradcam_video_paths, raw_video_paths


                  save_path = root_path.joinpath("DLIB_RESULT", target_name, target_task, target_fold_name, target_patient_id)
                  save_path.mkdir(parents = True, exist_ok = True)
                  for video_num in range(len(gradcam_video_paths)) : 
                      raw_video_path = raw_video_paths[video_num]
                      gradcam_video_path = gradcam_video_paths[video_num]
                      raw_video = np.load(raw_video_path)
                      gradcam_video = np.load(gradcam_video_path)

                      dlib_image_save_root = save_path.joinpath(raw_video_path.stem.replace('_raw', ''))
                      dlib_image_save_root.mkdir(exist_ok = True)
                      dlib_array_save_path = dlib_image_save_root.joinpath(raw_video_path.stem+'_DLIB').with_suffix('.npy')
                      dlib_gif_save_path = dlib_image_save_root.joinpath(raw_video_path.stem+'_DLIB').with_suffix('.gif')
                      dlib_gradcam_save_path = dlib_image_save_root.joinpath(raw_video_path.stem+'_GradCAM').with_suffix('.gif')

                      if dlib_gif_save_path.exists() : 
                          continue
                            
                      frames = []
                      points = []

                      cmap = plt.cm.get_cmap('RdPu')

                      for seq_len in range(raw_video.shape[0]) : 
                          seq_raw_image = raw_video[seq_len]
                          seq_raw_image = cv2.cvtColor(seq_raw_image, cv2.COLOR_BGR2RGB)
                          seq_gradcam_image = gradcam_video[seq_len] 
                          gray_raw_image = cv2.cvtColor(seq_raw_image ,cv2.COLOR_RGB2GRAY)
                          rects = dlib.get_frontal_face_detector()(gray_raw_image,1)

                          seq_raw_image = raw_video[seq_len]
                          seq_raw_image = cv2.cvtColor(seq_raw_image, cv2.COLOR_BGR2RGB)
                          seq_gradcam_image = gradcam_video[seq_len] 

                          seq_gradcam_image = cv2.resize(seq_gradcam_image, (224,224))
                          seq_gradcam_image = cmap(seq_gradcam_image)
                          seq_gradcam_image = np.uint8(seq_gradcam_image * 255)

                          gray_raw_image = cv2.cvtColor(seq_raw_image ,cv2.COLOR_RGB2GRAY)
                          rects = dlib.get_frontal_face_detector()(gray_raw_image,1)

                          rec_points = []
                          for rect in rects:
                              shape = predictor(gray_raw_image, rect)
                              for i in range(68):
                                  x = shape.part(i).x
                                  y = shape.part(i).y
                                  # Draw a circle around each facial landmark
                                  seq_gradcam_image = cv2.circle(seq_gradcam_image, (x, y), 2, (0, 0, 0), -1)
                                  rec_points.append([i, x, y])
                          points.append(rec_points) if len(rec_points) > 0 else points.append(np.ones(shape = (68,3)))
                          
                          image = Image.fromarray(seq_gradcam_image)
                          frames.append(image)

                          dlib_image_save_path = dlib_image_save_root.joinpath(f"SEQ_{seq_len:03}.jpg")
                          image.convert("RGB").save(dlib_image_save_path)
                      np.save(dlib_array_save_path.as_posix(), np.array(points))
                      frames[0].save(dlib_gif_save_path, format="GIF", append_images=frames, save_all=True, duration=100, loop=1)