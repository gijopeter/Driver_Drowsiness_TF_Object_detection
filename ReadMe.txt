
Dataset preparation:
The given dataset (Closed and Open eyes) is divided to train and validation dataset(90:10) by random selection. Training/Validation dataset were annotated
using LabelImg tool in the form of Pascal VOC files and converted to TFRecord file format, which was be used for model training.

TensorFlow 2 Detection Model Zoo provides a set of pretrained models(trained on COCO2017 dataset) and SSD MobileNet V2 FPNLite 640x640 was used for this
object detection task (transfer learning) becuase of its speed and better accuracy levels. Some of the pipeline configuration parameters were adapted 
for this task like num_classes, fine_tune_checkpoint_type andtrained for 50000 steps using google colab with GPU instance.

Tensorflow 2.0 does not support frozen graph any more, so save model is used for the generation of .pb file which was later used for inference.
https://stackoverflow.com/questions/55562078/tensorflow-2-0-frozen-graph-support

Two type of tagging was tried for annotation of open and closed eyes. Combined tagging of both eyes and individual tagging of eyes. 
Both the models were evaluated based on the Average precision and recall scores at various IOU which was generated from 10% reserved validation data set
(using model_main_tf2.py). The model which was trained on combined eye tagging found to pefrom better
Ref : https://blog.zenggyu.com/en/post/2018-12-16/an-introduction-to-evaluation-metrics-for-object-detection/
Moreover, the model inference was caried on all validation images and confidence score & tagging was verified manually. 

A python script is developed to acquire video and to run model inference on acquired frames. 
An audible drowsiness alert is generated if drivers eyes are closed for more than 5 seconds. 
Once a 'eye closed event' is detected, the closed event status is resetted only after an open eye is detected (considering safety)
(another additional implementation would be, to also reset the closed event status if no object detection happening for some predefined time)
At low light conditions the model generate some false Eye closed detection (large bounding boxes).
A filter logic based on the area of bouding box detection is introduced to avoid triggering of false drowsiness alert

Important: 
The training and validation data file size is so high to upload in upgrad portal. So please find the same in the following Gdrive link
(test.record, train.record)
https://drive.google.com/drive/folders/11FSFwSMAeRcfm5KpjbIIpK77bMGwkwiM?usp=sharing
