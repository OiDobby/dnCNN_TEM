# TEM_CNN_denoise

This code provides denoising method for TEM image using CNN.
We use density functional theory calculation to obtain the ground truth images.
For computational efficiency, we employ the pseudo-atomic orbital basis sets; OpenMX code (https://www.openmx-square.org/whatisopenmx.html).

We use the CNN model from the previous study (Phys. Rev. M 6, 123802 (2022), https://github.com/Fjoucken/Denoise_STM) with editing some options.

USAGE
1. generate disordered structures from "atomic_conf".
2. obtain electronic charge density in Gaussian cube format using OpenMX code (https://www.openmx-square.org/whatisopenmx.html).
3. get charge density 2D maps from electronic charge densities of disordered structures.
4. make corrupted images (training datasets) from "CNN_working_dir/Gen_DS". The charge density maps should be in "CNN_working_dir/Gen_DS/ori_png"
5. run "CNN_main.py" in CNN_working_dir (python3 CNN_main.py).
6. After training the CNN model, you can use plot.py and eval.py.
7. use the "plot.py" for plotting the prediction images, and use the "eval.py" for evaluating the prediction images (SSIM, MS-SSIM, PSNR)
8. The "plot.py" plots all the sliced patches, so you run the "img_merging.py". It makes the full-size image.

Our work reported in 
