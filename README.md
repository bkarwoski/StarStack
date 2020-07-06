# StarStack
This simple python script allows users to combine a large number of photos of the night sky, without the need for equatorial tracking equipment. By combining many short exposures together, background noise can be reduced.

Before and after: (100% zoom, left is one 5 second exposure, 50 mm F 1.8 len on a Sony Alpha 7 ii. Right is 120 exposures stacked.)
![before and after image stacking](before_after.jpg)
# Usage Directions
Place desired images to be stacked in a folder in the same directory as stack.py, labeled "jpg". Run stack.py. The image registration will take roughly 30 seconds per image.

# Results

![gif showing noise reduction](noise_reduction.gif)
## TODO
-Remove dual jpg and RAW dependency
-Gamma curve adjustment for brightening image