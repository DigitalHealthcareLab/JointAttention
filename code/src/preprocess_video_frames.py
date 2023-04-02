#%%
import os
from pathlib import Path
import myutils

PROJECT_PATH = Path(__file__).parents[1]
DATA_PATH = Path(PROJECT_PATH, "data")

sub_folder = "raw_data"
sub_folder_jpg = "raw_data_jpg"
path2aCatgs = Path(DATA_PATH, "raw_data")

#%%
listOfCategories = os.listdir(path2aCatgs)
listOfCategories, len(listOfCategories)

#%%
for cat in listOfCategories:
    print("category:", cat)
    path2acat = Path(path2aCatgs, cat)
    listOfSubs = os.listdir(path2acat)
    print("number of sub-folders:", len(listOfSubs))

#%%
extension = ".mp4"

for root, dirs, files in os.walk(path2aCatgs, topdown=False):
    for name in files:
        if extension not in name:
            continue
        path2vid = os.path.join(root, name)
        frames, vlen = myutils.get_frames(path2vid, n_frames=300)
        path2store = path2vid.replace(sub_folder, sub_folder_jpg)
        path2store = path2store.replace(extension, "")
        print(path2store)
        os.makedirs(path2store, exist_ok=True)
        myutils.store_frames(frames, path2store)
