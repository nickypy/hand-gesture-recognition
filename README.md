## Prerequisites
+ scikit-learn
+ opencv-python
+ keras
+ tensorflow
+ tensorflow-gpu (if a GPU is available)
+ numpy
+ Python 3.6.4+

## Building the dataset
```
$ python3 make_dataset.py
```
The `main_dir` variable must be edited depending on the actual path of the dataset.

Optionally, you can augment the images prior to making the dataset.
```
$ python3 augment-data.py
```

## Training the convnet
```
$ python3 tf-train.py
```

Note that this can take quite a long time without a GPU, espeically after augmenting the data. To alleviate this, the number of epochs can be decreased.


You can test the model independently by running
```
$ python3 test-data.py
```

## Running the webcam applet
```
python3 webcam.py
```
