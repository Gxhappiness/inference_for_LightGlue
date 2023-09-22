from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
import os
import glob
import time
from tqdm import tqdm
import pycolmap
import cv2
import numpy as np
np.set_printoptions(precision=8 , suppress=True) ## 8 decimal places are reserved, and Scientific notation is not used

data_root =  r'/the/path/to/data_root/' #### data root

#### Feature point extraction strategy
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
    txt_path = data_root + folders[i] + a + "pose_" + str(frame) + ".txt"
    # Check if the folder already exists, delete or skip if it exists
    if os.path.exists(txt_path):
        # os.remove(txt_path)
        continue
    rgb_path = data_root + folders[i] + '/images/'
    rgb_files = glob.glob(os.path.join(rgb_path, '*.jpg'))
    with open(txt_path, 'a') as f:
        f.write(rgb_files[0].split(".")[0].split("/")[-1] + " " + rgb_files[0].split(".")[0].split("/")[-1] + '\n' + "1.00000000 -0.00000000 0.00000000 0.00000000" + '\n' +
                "-0.00000000 1.00000000 0.00000000 -0.00000000" + '\n' +
                "-0.00000000 -0.00000000 1.00000000 0.00000000" + "\n" +
                "0.00000000 0.00000000 0.00000000 1.00000000" + "\n")
        for j in tqdm(range(0,len(rgb_files)-frame,frame), desc = folders[i]):
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

            K0 = pycolmap.infer_camera_from_image(rgb_files[j]).calibration_matrix()
            K1 = pycolmap.infer_camera_from_image(rgb_files[j+frame]).calibration_matrix()

            H, inliers = cv2.findHomography(points0.cpu().numpy(), points1.cpu().numpy(), cv2.RANSAC)
            # H, inliers = cv2.FindFundamentalMat(points0.numpy(), points1.numpy(), cv2.USAC_MAGSAC, 0.5, 0.999, 100000)

            ret = pycolmap.homography_decomposition(H, K0, K1, points0.cpu().numpy(), points1.cpu().numpy())

            R, t = ret['R'], ret['t']

            R_list = R.tolist()
            t_list = t.tolist()

            f.write(rgb_files[j].split(".")[0].split("/")[-1] + " " + rgb_files[j+frame].split(".")[0].split("/")[-1] + '\n')

            for k in range(3):
                R_list[k] = ['{:.8f}'.format(z) for z in R_list[k]]
                R_list[k].append('{:.8f}'.format(t_list[k]))
                f.write( " ".join(R_list[k]) + '\n')
            f.write("0.00000000 0.00000000 0.00000000 1.00000000" + '\n')