# <font color=red>Illumination</font> <font color=green>Adaptive</font> <font color=blue>Transformer</font> (IAT)

For Vision Tasks on both **Human Vision** :smile:  and **Machine Vision** :camera: 

**5 Tasks Under Various Lighting Conditions**: Low-light Enhancement ([LOL](https://daooshee.github.io/BMVC2018website/), [MIT5K](https://data.csail.mit.edu/graphics/fivek/)) // [Exposure Correction](https://github.com/mahmoudnafifi/Exposure_Correction) // [Low-Light Object Detection](https://arxiv.org/abs/1805.11227) // [Low-Light Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2021/html/Sakaridis_ACDC_The_Adverse_Conditions_Dataset_With_Correspondences_for_Semantic_Driving_ICCV_2021_paper.html) // [Various-Light Object Detection](https://bop.felk.cvut.cz/home/)

<!-- ![image](pics/WechatIMG416.png) -->
<div align="center">
  <img src="./pics/WechatIMG416.png" height="400">
</div>
<p align="center">
  Figure 1: IAT (illumination-adaptive-transformer) on multiply light vision challenges.
</p>

## Model Structure:


## Test and Train:


## Citation:

Detection and Segmentation are use [mmdetection](https://mmdetection.readthedocs.io/en/latest/) and [mmsegmentation](https://mmsegmentation.readthedocs.io/en/latest/), some of the code are also borrow from [Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE) and [Uniformer](https://github.com/Sense-X/UniFormer), thanks them both so much!

Citation of **Illumination Adaptive Transformer**:



We also have another work about to **low-light object detection**, **ICCV 2021**: Multitask AET with Orthogonal Tangent Regularity for Dark Object Detection [(code)](https://github.com/cuiziteng/ICCV_MAET) [(paper)](https://openaccess.thecvf.com/content/ICCV2021/papers/Cui_Multitask_AET_With_Orthogonal_Tangent_Regularity_for_Dark_Object_Detection_ICCV_2021_paper.pdf), please read check if you interest!

Citation of this work:

```
@InProceedings{Cui_2021_ICCV,
    author    = {Cui, Ziteng and Qi, Guo-Jun and Gu, Lin and You, Shaodi and Zhang, Zenghui and Harada, Tatsuya},
    title     = {Multitask AET With Orthogonal Tangent Regularity for Dark Object Detection},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {2553-2562}
}
```

