from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
import os
import glob
import time
from tqdm import tqdm
import numpy as np
np.set_printoptions(precision=8 , suppress=True) ## 8 decimal places are reserved, and Scientific notation is not used

data_root =  r'/the/path/to/data_root/' #### data root

##### Feature point extraction strategy
# method = "SuperPoint"
method = "DISK"

frame = 1 # frame(1,5,10,20,30). "frame = n" means n frame by frame calculation

##### single scene folder inference(If you only want to operate on a single scene folder, name it yourself)
# single , single_folder = False , ""
single , single_folder = True , "name it yourself"

GPU_id = 0  ## 0,1,2,3

def get_all_folders(path):
    folders = []
    for item in os.listdir(path):
        if os.path.isdir(os.path.join(path, item)):
            folders.append(item)
    return folders

folders = get_all_folders(data_root)

if method == "SuperPoint":
# SuperPoint+LightGlue
    extractor = SuperPoint(max_num_keypoints=1000).eval().cuda(GPU_id)  # load the extractor
    matcher = LightGlue(features='superpoint').eval().cuda(GPU_id)
    # extractor = SuperPoint(max_num_keypoints=None).eval().cuda(GPU_id)  # load the extractor
    # matcher = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).eval().cuda(GPU_id)
    a = "/SuperPoint_"
else:
# or DISK+LightGlue
#     extractor = DISK(max_num_keypoints=None).eval().cuda(GPU_id)  # load the extractor
#     matcher = LightGlue(features='disk', depth_confidence=-1, width_confidence=-1).eval().cuda(GPU_id)
    extractor = DISK(max_num_keypoints=1000).eval().cuda(GPU_id)  # load the extractor
    matcher = LightGlue(features='disk').eval().cuda(GPU_id)  # load the matcher
    a = "/DISK_"

for i in range(len(folders)):
    if single and folders[i] != single_folder:
        continue
    point_folder = data_root + folders[i] + a + "point_" + str(frame) + "/"
    # Check if the folder already exists, skip if it exists
    if not os.path.exists(point_folder):
        os.mkdir(point_folder)
        print(f"folder '{point_folder}' created")
    else:
        print(f"folder '{point_folder}' already exists")
        continue

    rgb_path = data_root + folders[i] + '/images/'
    rgb_files = glob.glob(os.path.join(rgb_path, '*.jpg'))
    for j in tqdm(range(0,len(rgb_files)-frame,frame), desc = folders[i]):
        point_txt = point_folder + rgb_files[j].split("/")[-1].split(".")[0] + "_" +\
                    rgb_files[j+frame].split("/")[-1].split(".")[0] + ".txt"
        # Check if the folder already exists, delete if it exists
        if os.path.exists(point_txt):
            os.remove(point_txt)
        time.sleep(0.05)
            # load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
        image0 = load_image(rgb_files[j]).cuda(GPU_id)
        image1 = load_image(rgb_files[j+frame]).cuda(GPU_id)

            # extract local features
        feats0 = extractor.extract(image0)  # auto-resize the image, disable with resize=None
        feats1 = extractor.extract(image1)

            # match the features
        matches01 = matcher({'image0': feats0, 'image1': feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
        matches = matches01['matches']  # indices with shape (K,2)
        points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
        points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)

        p0 = points0.tolist()[:-1]
        p1 = points1.tolist()[:-1]
        for k in range(len(p0)):
            p0[k] = [str(round(z0)) for z0 in p0[k]]
            p1[k] = [str(round(z1)) for z1 in p1[k]]
        p = [" ".join(p0[i]+p1[i]) for i in range(len(p0))]
        with open(point_txt, 'a') as f:
            f.write("\n".join(p))



