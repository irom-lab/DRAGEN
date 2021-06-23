# DRAGEN - Distributionally Robust Policy Learning via Adversarial Environment Generation

<!-- [Paper](https://arxiv.org/abs/2008.01913) | [Review](https://drive.google.com/file/d/1VmLh07UuOVhDxGXh2YoVCJf3GvHNbG0M/view?usp=sharing) | [Experiment video](https://www.youtube.com/watch?v=dfXyHvOTolc&t=3s) | [5min presentation at CoRL 2020](https://www.youtube.com/watch?v=nabtvOWoIlo&feature=emb_logo) -->

<!-- [![Watch the video](https://img.youtube.com/vi/dfXyHvOTolc/maxresdefault.jpg)](https://www.youtube.com/watch?v=dfXyHvOTolc) -->

This repository includes codes for synthetic trainings of the two robotic tasks in the paper:
1. Swinging up a pendulum with onboard vision
2. Grasping realistic 2D/3D grasping

Although the codes for all experiments are included here, we only provide instructions for re-producing pendulum experiments here. Re-producing grasping experiments relies on several custom packages (SDF sampling, mesh processing, off-screen rendering) that requires careful installation; we are happy to provide instructions upon further request. We do provide the instructions for running grasping policy training in 2D setting (with PyBullet simulation).

### Dependencies (`pip install` with python=3.7):
1. pybullet
PIL, torch, torchvision, gym, psutil, mnist, h5py, rlpyt (own fork)
perlin_noise, shapely, trimesh

### File naming:

### Running pendulum experiments:
1. Generate
2. 
<!-- (**Note:** the default number of -->

### Running grasping policy training in 2D setting:
1. Generate training and testing objects by running `generate_2d_shapes.py`.
2. Specify object path in ```train_grasp.py``` and run it. 

<!-- ### Future release
1. 
