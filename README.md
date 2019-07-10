## Description
This repo contains some of my dissertation work for the MPhil in Machine Learning at the University of Cambridge. I'm working on Interpretable Machine Learning. The original repo is from [TCAV](https://github.com/tensorflow/tcav). This is currently a work in progress!

## Visualizing CAVs
CAVs, or concept activation vectors are training using example images that contain a particular concept of interest. Below are some example concepts for the *striped* concept class.

<img src="concept_examples/striped_1.jpg" height="100"><img src="concept_examples/striped_2.jpg" height="100"><img src="concept_examples/striped_3.jpg" height="100">

To learn the concept, a set of *random images* (from ImageNet) are used in a binary classfication task.

<img src="concept_examples/random_1.JPEG" height="100"><img src="concept_examples/random_2.JPEG" height="100">
<img src="concept_examples/random_3.JPEG" height="100">

The CAV is normal to the hyperplane that separates these two classes, learned through a linear classifier. Part of my work is visualizing the two classes and learned CAV. Through PCA, I have visualized CAVs under several conditions.

normal

<img src="figs_for_github/pca_striped_sub_1-random500_0-mixed9-linear-0.1.png" height="300">

diff sub 1

<img src="figs_for_github/pca_striped_sub_2-random500_0-mixed9-linear-0.1.png" height="300">

noise
<img src="figs_for_github/pca_striped_sub_1-noise_color-mixed9-linear-0.1.png" height="300">




