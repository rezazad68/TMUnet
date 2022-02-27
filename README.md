# [Contextual Attention Network: Transformer Meets U-Net](http://openaccess.thecvf.com/content_ICCVW_2019/papers/VRMI/Azad_Bi-Directional_ConvLSTM_U-Net_with_Densley_Connected_Convolutions_ICCVW_2019_paper.pdf)

Contexual attention network for medical image segmentation with state of the art results on skin lesion segmentation, multiple myeloma cell segmentation. This method incorpotrates the transformer module into a U-Net structure so as to concomitantly capture long-range dependancy along with resplendent local informations.
If this code helps with your research please consider citing the following paper:
</br>
> [R. Azad](https://scholar.google.com/citations?hl=en&user=Qb5ildMAAAAJ&view_op=list_works&sortby=pubdate), [Moein Heidari](https://scholar.google.com/citations?hl=en&user=CUHdgPcAAAAJ&view_op=list_works&sortby=pubdate), [Yuli Wu](https://scholar.google.com/citations?user=qlun0AgAAAAJ) and [Dorit Merhof
](https://scholar.google.com/citations?user=JH5HObAAAAAJ&sortby=pubdate), "Contextual Attention Network: Transformer Meets U-Net", MICCAI, 2022, download [link](https://arxiv.org/pdf/1909.00166.pdf).



#### Please consider starring us, if you found it useful. Thanks

## Updates
- February 27, 2022: First release (Complete implemenation for [SKin Lesion Segmentation on ISIC 2017](https://challenge.isic-archive.com/landing/2017/), [SKin Lesion Segmentation on ISIC 2018](https://challenge2018.isic-archive.com/), [SKin Lesion Segmentation on PH2](https://www.kaggle.com/synked/ph2-modified) and [Multiple Myeloma Cell Segmentation (SegPC 2021)](https://www.kaggle.com/sbilab/segpc2021dataset) dataset added.)
- February 26, 2022: Paper sent to the [MICCAI Conference](https://conferences.miccai.org/2022]) 2022.

This code has been implemented in python language using Pytorch library and tested in ubuntu OS, though should be compatible with related environment. following Environement and Library needed to run the code:

- Python 3
- Pytorch


## Run Demo
For training deep model and evaluating on each data set follow the bellow steps:</br>
1- Download the ISIC 2018 train dataset from [this](https://challenge.isic-archive.com/data) link and extract both training dataset and ground truth folders inside the `dataset_isic18`. </br>
2- Run `Prepare_ISIC2018.py` for data preperation and dividing data to train,validation and test sets. </br>
3- Run `Train_Skin_Lesion_Segmentation.py` for training the model using trainng and validation sets. The model will be train for 100 epochs and it will save the best weights for the valiation set. </br>
4- For performance calculation and producing segmentation result, run `Evaluate_Skin.py`. It will represent performance measures and will saves related results in `output` folder.</br>

**Notice:**
For training and evaluating on ISIC 2017 and ph2 follow the bellow steps: :</br>
**ISIC 2017**- Download the ISIC 2017 train dataset from [this](https://challenge.isic-archive.com/data) link and extract both training dataset and ground truth folders inside the `dataset_isic18\7`. </br> then Run ` 	Prepare_ISIC2017.py` for data preperation and dividing data to train,validation and test sets. </br>
**ph2**- Download the ph2 dataset from [this](https://www.dropbox.com/s/k88qukc20ljnbuo/PH2Dataset.rar) link and extract it then Run ` 	Prepare_ph2.py` for data preperation and dividing data to train,validation and test sets. </br>
Follow step 3 and 4 for model traing and performance estimation. For ph2 dataset you need to first train the model with ISIC 2018 data set and then fine-tune the trained model using ph2 dataset.



## Quick Overview
![Diagram of the proposed method](https://github.com/rezazad68/TMUnet/blob/main/Figures/Method%20(Main).png)

### Perceptual visualization of the proposed Contextual Attention module.
![Diagram of the proposed method](https://github.com/rezazad68/TMUnet/blob/main/Figures/Method%20(Submodule).png)


## Results
For evaluating the performance of the proposed method, Two challenging task in medical image segmentaion has been considered. In bellow, results of the proposed approach illustrated.
</br>
#### Task 1: SKin Lesion Segmentation


#### Performance Comparision on SKin Lesion Segmentation
In order to compare the proposed method with state of the art appraoches on SKin Lesion Segmentation, we considered Drive dataset.  

Methods | Year |F1-scores | Sensivity| Specificaty| Accuracy | AUC
------------ | -------------|----|-----------------|----|---- |---- 
Chen etc. all [Hybrid Features](https://link.springer.com/article/10.1007/s00138-014-0638-x)        |2014	  |	-       |0.7252	  |0.9798	  |0.9474	  |0.9648
Azzopardi  et. all [Trainable COSFIRE filters ](https://www.sciencedirect.com/science/article/abs/pii/S1361841514001364)   |2015	  |	-       |0.7655	  |0.9704	  |0.9442	  |0.9614
Roychowdhury and et. all [Three Stage Filtering](https://ieeexplore.ieee.org/document/6848752)|2016 	|	-       |0.7250	  |**0.9830**	  |0.9520	  |0.9620
Liskowsk  etc. all[Deep Model](https://ieeexplore.ieee.org/document/7440871)	  |2016	  |	-       |0.7763	  |0.9768	  |0.9495	  |0.9720
Qiaoliang  et. all [Cross-Modality Learning Approach](https://ieeexplore.ieee.org/document/7161344)|2016	  |	-       |0.7569	  |0.9816	  |0.9527	  |0.9738
Ronneberger and et. all [U-net](https://arxiv.org/abs/1505.04597)	     	    |2015   | 0.8142	|0.7537	  |0.9820	  |0.9531   |0.9755
Alom  etc. all [Recurrent Residual U-net](https://arxiv.org/abs/1802.06955)	|2018	  | 0.8149  |0.7726	  |0.9820	  |0.9553	  |0.9779
Oktay  et. all [Attention U-net](https://arxiv.org/abs/1804.03999)	|2018	  | 0.8155	|0.7751	  |0.9816	  |0.9556	  |0.9782
Alom  et. all [R2U-Net](https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf)	        |2018	  | 0.8171	|0.7792	  |0.9813	  |0.9556	  |0.9784
Azad et. all [Proposed BCDU-Net](https://github.com/rezazad68/LSTM-U-net/edit/master/README.md)	  |2019 	| **0.8222**	|**0.8012**	  |0.9784	  |**0.9559**	  |**0.9788**


#### SKin Lesion Segmentation segmentation result on test data

![SKin Lesion Segmentation  result](https://github.com/rezazad68/TMUnet/blob/main/Figures/Skin%20lesion_segmentation.png)
(a) Input images. (b) Ground truth. (c) [U-net](https://arxiv.org/abs/2102.10662). (d) [Gated Axial-Attention](https://arxiv.org/abs/2102.10662). (e) Proposed method without a contextual attention module and (f) Proposed method.


## Multiple Myeloma Cell Segmentation

#### Performance Evalution on the Multiple Myeloma Cell Segmentation task

Methods | Year |F1-scores | Sensivity| Specificaty| Accuracy | PC | JS 
------------ | -------------|----|-----------------|----|---- |---- |---- 
Ronneberger and etc. all [U-net](https://arxiv.org/abs/1505.04597)	     	    |2015   | 0.647	|0.708	  |0.964	  |0.890  |0.779 |0.549
Alom  et. all [Recurrent Residual U-net](https://arxiv.org/abs/1802.06955)	|2018	  | 0.679 |0.792 |0.928 |0.880	  |0.741	  |0.581
Oktay  et. all [Attention U-net](https://arxiv.org/abs/1804.03999)	|2018	  | 0.665	|0.717	  |0.967	  |0.897	  |0.787 | 0.566 
Alom  et. all [R2U-Net](https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf)	        |2018	  | 0.691	|0.726	  |0.971	  |0.904	  |0.822 | 0.592
Azad et. all [Proposed BCDU-Net](https://github.com/rezazad68/LSTM-U-net/edit/master/README.md)	  |2019 	| **0.847**	|**0.783**	  |**0.980**	  |**0.936**	  |**0.922**| **0.936**
Azad et. all [MCGU-Net](https://128.84.21.199/pdf/2003.05056.pdf)	  |2020	| **0.895**	|**0.848**	  |**0.986**	  |**0.955**	  |**0.947**| **0.955**




#### Multiple Myeloma Cell Segmentation results

![Multiple Myeloma Cell Segmentation result](https://github.com/rezazad68/TMUnet/blob/main/Figures/Cell_segmentation.png)


### Model weights
You can download the learned weights for each task in the following table. 

Task | Dataset |Learned weights
------------ | -------------|----
Skin Lesion Segmentation | [Drive](http://www.isi.uu.nl/Research/Databases/DRIVE/) |[BCDU_net_D3](https://drive.google.com/open?id=1_hpfspGGJcWyFcGLXkFUa4k1NdUyOSOb)
Multiple Myeloma Cell Segmentation | [ISIC2018](https://challenge.kitware.com/#phase/5abcb19a56357d0139260e53) |[BCDU_net_D3](https://drive.google.com/open?id=1EPRC-YmMk0AjHbdjoVy53jlSuweSbAHX)



### Query
All implementations are done by Reza Azad and Moein Heidari. For any query please contact us for more information.

```python
rezazad68@gmail.com
moeinheidari7829@gmail.com

```
