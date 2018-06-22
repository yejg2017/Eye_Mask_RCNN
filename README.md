# Eye_Mask_RCNN
Mask RCNN used for  Eye dataset.Eye dataset annotation use VGG Image Annotator.

This is an implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) on Python 3, Keras, and TensorFlow. 
The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature
Pyramid Network (FPN) and a ResNet101 backbone.

The repository includes:
* Source code of Mask R-CNN built on FPN and ResNet101.
* Training code for MS COCO
* Pre-trained weights for MS COCO
* Jupyter notebooks to visualize the detection pipeline at every step
* ParallelModel class for multi-GPU training
* Evaluation on MS COCO metrics (AP)
* Example of training on your own dataset


The code is documented and designed to be easy to extend. If you use it in your research, please consider referencing this repository. 
If you work on 3D vision, you might find our recently released [Matterport3D](https://matterport.com/blog/2017/09/20/announcing-matterport3d-research-dataset/) dataset 
useful as well.This dataset was created from 3D-reconstructed spaces captured by our customers who agreed to make them publicly available 
for academic use. You can see more examples [here](https://matterport.com/gallery/).



### Getting Started
* [demo.ipynb](samples/demo.ipynb) Is the easiest way to start. It shows an example of using a model pre-trained on MS COCO to segment objects in your own images.
It includes code to run object detection and instance segmentation on arbitrary images.

* [train_shapes.ipynb](samples/shapes/train_shapes.ipynb) shows how to train Mask R-CNN on your own dataset. This notebook introduces a toy dataset (Shapes) to demonstrate training on a new dataset.

* ([model.py](mrcnn/model.py), [utils.py](mrcnn/utils.py), [config.py](mrcnn/config.py)): These files contain the main Mask RCNN implementation. 


* [inspect_data.ipynb](samples/coco/inspect_data.ipynb). This notebook visualizes the different pre-processing steps
to prepare the training data.

* [inspect_model.ipynb](samples/coco/inspect_model.ipynb) This notebook goes in depth into the steps performed to detect and segment objects. It provides visualizations of every step of the pipeline.

* [inspect_weights.ipynb](samples/coco/inspect_weights.ipynb) This notebooks inspects the weights of a trained model and looks for anomalies and odd patterns.

* [Eye.ipynb](samples/shapes/Eye.ipynb)  This notebook introduce how to use own Eye dataset to train own model based on coco pre-trained model.the dataset build on [VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/via.html).It is easy to use it(标签自己的数据.( [balloon.py](samples/balloon/balloon.py),千万要看这个代码，要不然作死)

* [coco_weight.h5](https://drive.google.com/open?id=1jHRnKPNFQdZvWqplBljugazWoJrA78l0)  the pre_trained coco mask rcnn model weight can be downloaded from here.(可以从这里下载与训练模型coco_mask.h5)


### 其他的
 参考 https://github.com/yejg2017/Mask_RCNN
 
 各种RCNN的简单理解:  https://www.cnblogs.com/skyfsm/p/6806246.html
