import os
import imageio
import cv2

frame_dir = './to_gif/'
images = []
for idx, file_name in enumerate(sorted(os.listdir(frame_dir))):
    if file_name.endswith('.jpg'):
        # print(file_name)
        file_path = os.path.join(frame_dir, file_name)
        img = cv2.imread(file_path)
        text = "Frames: " + str((idx + 1) ** 2)
        img = cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255, 255), 2)
        images.append(img)
imageio.mimsave('noise_reduction.gif', images, duration=1.0)