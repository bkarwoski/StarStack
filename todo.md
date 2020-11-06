# Future work for the project
-Implement Poetry / venv management
-Add tool for batch cropping
-Command line args for picture locations?
-remove dependency for .ARW and .jpg images
-Separate gif creation script
-Make functions called more modular
-Find good parameters for RAW processing

# RAW processing notes
Testing on Saguaro desert images.

Plain defaults on rawpy postprocess: 0.9108, 0.9104
no_auto_bright=True: 0.9418, 0.9416
defaults, gaussian blur(img, (0,0), 3): 0.9877, 0.9876
defaults, gaussian blur(img, (0,0), 5): 0.9949, 0.9949
defaults, gaussian blur(img, (0,0), 10): 0.9983, 0.9983
defaults, gaussian blur(img, (0,0), 20): 0.9991, 0.9992
defaults, gaussian blur(img, (0,0), 40): 0.9993, 0.9994
defaults, gaussian blur(img, (0,0), 80): 0.9993, 0.9992
//Could this be a red Herring? Maybe Blurred Images are always better scores, whether or not the alignment is any better?

The gaussian blur always improves the ECC score.

