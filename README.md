# Medical image registration for CT/X-ray
2D/3D registration between CT/X-ray images

![framework](/img/framework.png " ")

In this [paper](https://arxiv.org/abs/2305.06252), we propose a novel two-stage 2D/3D registration framework,
Embedded Feature Similarity Optimization with Specific Parameter Initialization (SOPI), which can align the images automatically without a large amount
of real X-ray data for training and weaken the effect of incorrect initialization on
the registration algorithm. In this framework, we propose a regressive parameterspecific module, Rigid Transformation Parameter Initialization (RTPI) module,
to initialize pose parameter and an iterative fine-registration network to align the
two images precisely by using embedded features. The framework estimates the
transformation parameter that best aligns two images using one intra-operative
x-ray and one pre-operative CT as input.
![RTPI](/img/RTPI.png " ")
![encoder](/img/composite_encoder.png " ")

(PS: The CPU-Net file is some early immature ideas, it does not have much value for reference)
