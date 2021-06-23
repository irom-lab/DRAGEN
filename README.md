# DRAGEN - Distributionally Robust Policy Learning via Adversarial Environment Generation

<!-- [Paper](https://arxiv.org/abs/2008.01913) | [Review](https://drive.google.com/file/d/1VmLh07UuOVhDxGXh2YoVCJf3GvHNbG0M/view?usp=sharing) | [Experiment video](https://www.youtube.com/watch?v=dfXyHvOTolc&t=3s) | [5min presentation at CoRL 2020](https://www.youtube.com/watch?v=nabtvOWoIlo&feature=emb_logo) -->

<!-- [![Watch the video](https://img.youtube.com/vi/dfXyHvOTolc/maxresdefault.jpg)](https://www.youtube.com/watch?v=dfXyHvOTolc) -->

This repository includes codes for synthetic trainings of the two robotic tasks in the paper:
1. Swinging up a pendulum with onboard vision
2. Grasping realistic 2D/3D grasping

Although the codes for all experiments are included here, we only provide instructions for re-producing pendulum experiments here. Re-producing grasping experiments relies on several custom packages (SDF sampling, mesh processing, off-screen rendering) that requires careful installation; we are happy to provide instructions upon further request (but would be difficult to keep anonymity). We do provide the instructions for running grasping policy training in 2D setting (with PyBullet simulation).

### Dependencies:
`pip install` with python=3.7/3.8: pybullet, Pillow, torch, torchvision, gym, psutil, python-mnist, h5py, pyyaml, perlin_noise (pendulum baseline); shapely, trimesh (grasping) \
`pip install -e .` to install locally: rlpyt folder included in the supplementary materials, which is a custom fork of [rlpyt](https://github.com/astooke/rlpyt) library for SAC training. Running pendulum experiments requires a GPU.

### File naming:
`ae` is for DRAGEN training; `dr` is for baselines (noise, domain randomization, etc).

### Running pendulum experiments:
1. (For Digit) install and download MNIST dataset [here](https://pypi.org/project/python-mnist/) and download USPS dataset [here](https://www.kaggle.com/bistaumanga/usps-dataset); (for Urban) download Apollo Synthetic dataset [here](https://apollo.auto/synthtic.html).
2. Generate Landmark/Digit/Urban images by running ```generate_landmark.py``` and ```process_mnist/usps/apollo.py```.
3. Specify dataset paths in provided config files.
4. Run ```python run_pen_ae.py {config filename}```, or ```python run_pen_dr.py {config filename}``` for baselines. 
<!-- (**Note:** the default number of -->

### Running grasping policy training in 2D setting:
1. Generate training and testing objects by running `generate_2d_shapes.py`.
2. Specify object path in ```train_grasp.py``` and run it. 

<!-- ### Future release
1. 
