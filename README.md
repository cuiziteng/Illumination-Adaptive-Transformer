# <font color=red>Illumination</font> <font color=green>Adaptive</font> <font color=blue>Transformer</font> (IAT) [(paper link)](https://arxiv.org/abs/2205.14871) 

For Vision Tasks under various lighting conditions, towards both **Human Vision** :smile:  and **Machine Vision** :camera: 

**5 Tasks Under Various Lighting Conditions**: 1. Low-light Enhancement ([LOL](https://daooshee.github.io/BMVC2018website/), [MIT5K](https://data.csail.mit.edu/graphics/fivek/)) // 2. [Exposure Correction](https://github.com/mahmoudnafifi/Exposure_Correction) // 3. [Low-Light Object Detection](https://arxiv.org/abs/1805.11227) // 4. [Low-Light Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2021/html/Sakaridis_ACDC_The_Adverse_Conditions_Dataset_With_Correspondences_for_Semantic_Driving_ICCV_2021_paper.html) // 5. [Various-Light Object Detection](https://bop.felk.cvut.cz/home/)

<!-- ![image](pics/WechatIMG416.png) -->
<div align="center">
  <img src="./pics/WechatIMG416.png" height="400">
</div>
<p align="center">
  Figure 1: IAT (illumination-adaptive-transformer) for multi light conditions vision tasks.
</p>


## Model Structure:


## Test and Train:


## Citation:

Detection and Segmentation are use [mmdetection](https://mmdetection.readthedocs.io/en/latest/) and [mmsegmentation](https://mmsegmentation.readthedocs.io/en/latest/), some of the code are borrow from [Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE) and [UniFormer](https://github.com/Sense-X/UniFormer), thanks them both so much!

Citation:

‘’‘
@misc{Illumination_Adaptive_Transformer,
  doi = {10.48550/ARXIV.2205.14871},
  url = {https://arxiv.org/abs/2205.14871},
  author = {Cui, Ziteng and Li, Kunchang and Gu, Lin and Su, Shenghan and Gao, Peng and Jiang, Zhengkai and Qiao, Yu and Harada, Tatsuya},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Illumination Adaptive Transformer},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
’‘’

We also have another work about to **low-light object detection**, **ICCV 2021**: Multitask AET with Orthogonal Tangent Regularity for Dark Object Detection [(code)](https://github.com/cuiziteng/ICCV_MAET) [(paper)](https://openaccess.thecvf.com/content/ICCV2021/papers/Cui_Multitask_AET_With_Orthogonal_Tangent_Regularity_for_Dark_Object_Detection_ICCV_2021_paper.pdf), please read if you interest!

