# DRAGEN - Distributionally Robust Policy Learning via Adversarial Environment Generation

<!-- [Paper](https://arxiv.org/abs/2008.01913) | [Review](https://drive.google.com/file/d/1VmLh07UuOVhDxGXh2YoVCJf3GvHNbG0M/view?usp=sharing) | [Experiment video](https://www.youtube.com/watch?v=dfXyHvOTolc&t=3s) | [5min presentation at CoRL 2020](https://www.youtube.com/watch?v=nabtvOWoIlo&feature=emb_logo) -->

<!-- [![Watch the video](https://img.youtube.com/vi/dfXyHvOTolc/maxresdefault.jpg)](https://www.youtube.com/watch?v=dfXyHvOTolc) -->

This repository includes codes for synthetic trainings of these robotic tasks in the paper:
1. Grasping diverse mugs
2. Planar box pushing using visual-feedback
3. Vision-based navigation through home environments

Although the codes for all examples are included here, only the pushing example can be run without any additional codes/resources. The other two examples require data from online object dataset and object post-processing, which can take significant amount of time to set up and involves licensing. Meanwhile, all objects (rectangular boxes) used for the pushing example can be generated through URDF files (`generativeBox.py`).

Moreover, we provide the pre-trained weights for the decoder network of the cVAE for the pushing example. The posterior policy distribution can be trained then using the weights and the prior distribution (unit Gaussians).

### Dependencies (`pip install` with python=3.7):
1. pybullet

### File naming:

### Running the pushing example:
1. Generate a large number (3000) of boxes by running ```python generateBox.py --obj_folder=...``` and specifying the path to the object URDF files generated.
2. Modify ```obj_folder``` in `push_pac_easy.json` and `push_pac_hard.json`
3. Train pushing tasks ("Easy" or "Hard" difficulty) by running ```python trainPush_es.py push_pac_easy``` (or `hard`). The final bound is also computed by specifying `L` (number of policies sampled for each environment for computing the sample convergence bound) in the json file. (**Note:** the default number of

<!-- ### Future release
1. 

(**Note:** we do not plan to release instructions to replicate results of the indoor navigation example in the near future. We plan to refine the simulation in a future version of the paper.)

### Omission of the paper
1. We made the assumption that the latent variables in the CVAE are independent of the states (p(z|s_{1:T}) = p(z)). -->
