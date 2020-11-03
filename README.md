# Image Captioning Studies

The aim of this project is to create a test-bed for implementing and evaluating various (visual, multimodal, attention-based etc.) state-of-the-art image captioning machine learning algorithms. For a comprehensive review of the field until circa 2018, please refer to [this](https://arxiv.org/pdf/1810.04020.pdf) paper. We use the [tensorflow tutorial on image captioning](https://www.tensorflow.org/tutorials/text/image_captioning) as the starting point for code development. 

## Baseline model
Please refer to the Jupyter notebook therein for further details on the baseline model. Here we only briefly summarize the key features.

* Model: Attention-based supervised learning similar to [Xu et al.](https://arxiv.org/pdf/1502.03044.pdf)
  * Image encoder: Last CNN layer from InceptionV3 trained on ImageNet
  * Language model: Gated recurrent unit (GRU) with 512 units
  * Attention model: Based on [Bahdanau](https://arxiv.org/pdf/1409.0473.pdf) 
* Dataset: 6 K [images](http://images.cocodataset.org/zips/train2014.zip) with [annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip) from MS-COCO
* Training-validation split: 80-20
* Batch-size: 64
* Epochs: 10
* Activations: 
  * Images: RELU
  * Language: tanh
  



