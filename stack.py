#!/usr/bin/env python3
from util import *
from datetime import datetime 
import glob
import os
import pathlib
import sys

def stack(jpgs_path, raw_path):
    jpg_list = sorted(glob.glob(jpgs_path))
    raw_list = sorted(glob.glob(raw_path))
    imgs_to_stack = slice(0,2)
    jpg_list = jpg_list[imgs_to_stack]
    raw_list = raw_list[imgs_to_stack]
    trans_prior = np.eye(3, 3, dtype=np.float32) #todo should be int?
    init_jpg = load_image(jpg_list[0])
    init_raw = load_image(raw_list[0])
    img_stack = init_raw.astype(np.float32)
    denoise_idxs = []
    # denoise_idxs = [0, 1, 3, 7, 15, 31, 63, 119]
    print("Going to stack", len(jpg_list), "images.")
    now = datetime.now()
    out_dir = os.path.join("out", now.strftime("%Y-%m-%d-%H-%M-%S"))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    jpg_list = jpg_list[1:] #remove first value. It's already on the image stack
    for idx, fname in enumerate(jpg_list):
        img = load_image(fname)
        raw = load_image(raw_list[idx])
        transform, cc = ecc_transform(init_jpg, img, prior=trans_prior)
        trans_prior = transform
        frame_dim = (init_raw.shape[1], init_raw.shape[0])
        img_warp = cv2.warpPerspective(raw, transform, frame_dim,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
        flags=cv2.WARP_INVERSE_MAP)
        img_stack += img_warp
        #todo remove denoise gif code from here
        if idx in denoise_idxs:
            if not os.path.exists(os.path.join(out_dir, "to_gif")):
                os.makedirs(os.path.join(out_dir, "to_gif"))
            center = (img_stack.shape[0] // 2, img_stack.shape[1] // 2)
            h = 200
            w = 300
            img_out = img_stack[center[0] - h:center[0] + h,
                                center[1] - w:center[1] + w] / (idx + 1)
            out = 'to_gif/frame_' + str(idx + 1).zfill(3) + '.jpg'
            print(out, "saved") 
            cv2.imwrite(os.path.join(out_dir, out), img_out)
        print(fname, "stacked, frame", str(idx + 2), "of",
              str(len(jpg_list)), "correlation coefficient: ", f"{cc:.4}")

    img_stack /= len(jpg_list)
    stars, background = rm_background(img_stack, ratio=0)
    np.save(os.path.join(out_dir, "stars.npy"), stars)
    np.save(os.path.join(out_dir, "background.npy"), background)
    stars = cv2.cvtColor(stars, cv2.COLOR_BGR2RGB)
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    now = datetime.now()
    out_name = "out.jpg"
    cv2.imwrite(os.path.join(out_dir, out_name), stars)
    cv2.imwrite(os.path.join(out_dir, "removed_background.jpg"), background)
    print(out_name, "saved")

if __name__ == "__main__":
    jpgs_path = "/home/blake/Pictures/Sand_Dune_Stars/time_lapse/jpg/*"
    raw_path = "/home/blake/Pictures/Sand_Dune_Stars/time_lapse/raw/*"
    stack(jpgs_path, raw_path)
