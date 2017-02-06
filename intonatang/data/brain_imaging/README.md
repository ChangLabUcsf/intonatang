# Brain imaging data

The brain imaging data for each subject consists of one .png image and one .mat file. The 
ECXXX_brain2D.png image is a lateral view of a 3D brain, reconstructed from an MRI scan. The 
ECXXX_elec_pos2D.mat files contain a variable xy, which is a 2 x 256 ndarray containing the 
locations of each electrode on the 2D image. The locations are given by x and y pixel values.
