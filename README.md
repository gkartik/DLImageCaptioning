# Image Captioning Studies

The aim of this project is to create a test-bed for implementing and evaluating various (visual, multimodal, attention-based etc.) state-of-the-art image captioning machine learning algorithms. For a comprehensive review of the field until circa 2018, please refer to [this](https://arxiv.org/pdf/1810.04020.pdf) paper. We use the [tensorflow tutorial on image captioning](https://www.tensorflow.org/tutorials/text/image_captioning) as the starting point for code development. 

## Baseline model
Please refer to the Jupyter notebook therein for further details on the baseline model. Here we only briefly summarize the key features.

* Model: Attention-based supervised learning similar to [Xu et al.](https://arxiv.org/pdf/1502.03044.pdf)
  * Image encoder: Last CNN layer from InceptionV3 trained on ImageNet
  * Language model: Gated recurrent unit (GRU) with 512 units
  * Attention model: Based on [Bahdanau](https://arxiv.org/pdf/1409.0473.pdf)
* Loss:[SparseCategoricalCrossEntropy](https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy) 
* Dataset: 6 K [images](http://images.cocodataset.org/zips/train2014.zip) with [annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip) from MS-COCO
* Training-validation split: 80-20
* Batch-size: 64
* Epochs: 10
* Activations: 
  * Images: RELU
  * Language: tanh
  
## Case 1: Comparison of GRU and Simple (Elman) RNN

1. We made the following changes w.r.t baseline:

```
self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
```
replaced by 
```
self.myRNN = tf.keras.layers.SimpleRNN(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
```
and 
```
 output, state = self.gru(x)
```
by
```
 output, state = self.myRNN(x)
```
2. We also turned off random shuffling of training samples before running the two models to ensure identical training sets i.e.,

```
#random.shuffle(img_paths)
#random.shuffle(img_keys)
```

### Results

The plot below shows the comparison of training loss by epochs for simple RNN and GRU. We note that by 10 epochs both models converge to the same loss.

![Image of loss comparison](https://github.com/gkartik/DLImageCaptioning/blob/main/TrainingLossComparison.png)


