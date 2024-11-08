
import numpy as np

xyz = np.random.rand(50000, 3) * 100
voxel_size = 0.5 # each voxel will be half of the pointclouds unit along each axis

xyz_q = np.round(np.array(xyz/voxel_size)).astype(int) # quantized point values, here you will loose precision

vox_grid = np.zeros((int(100/voxel_size)+1, int(100/voxel_size)+1, int(100/voxel_size)+1)) #Empty voxel grid

vox_grid[xyz_q[:,0],xyz_q[:,1],xyz_q[:,2]] = 1 # Setting all voxels containitn a points equal to 1

xyz_v = np.asarray(np.where(vox_grid == 1)) # get back indexes of populated voxels

print(xyz_v)
