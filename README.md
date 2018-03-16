# Facial Expression Recognition
A real-time facial expression recognition system based on opencv and CNN.

## Abstract
In this project, we implemented a real time expression resognition system using webcam streaming and CNN. Also, we use pytorch, implemented CNN, RNN, ResNet, and compared their training and testing accuracy. Analysed their advantages.

## Dataset
We are using [fer2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data), which is anounced in Kaggle competition in 2013.

## Environment

### HardWare

CPU : 3.1 GHz Intel Core i7

GPU : nvidia GTX 960 2G

RAM : 16G

Strongly recommend using ['Anaconda'](https://www.continuum.io/downloads).

Create virtual environment using anaconda

```
conda create -n myenv python=3.4
conda info --envs
```

Set my virtual environment name to myenv

Activate virtual environment
```
source activate env-name
conda install scikit-learn
conda install opencv
pip install --upgrade keras
```
`h5py` is used for saving weights of pre-trained models.
```
conda install h5py
```
when finished, simply type:
```
source deactivate 
```

## Usage
### Facial expression detection
After installing all the dependencies, you can direct to `webcam` directory, and simply run 
```
python webcam_detection.py
```
If everything went well, our system should start detecting users' expression and show the result on screen. Also, there will be a history log on your terminal. If you will. you could save those results for later use by typing this command:
```
python webcam_detection.py > log.txt
```
which coould save those result to log.txt file.

### Facial expression detection - single image
We also support determination of signle image, simply using the following command:
```
python webcam_detection.py --testImage images.jepg
```
This could show the result on terminal and it's probability.

### Facial expression detection - images
Simply store your images in a directory, and use the following command:
```
python webcam_detection.py --dataset dir_name
```
This could show the resulsts on terminal and their probabilities.

## Pytorch Neural Network comparison
In this part, you can simply direct to `model_pytorch` directory. There are several networks implemented by ourself. We are trying to find out the accuracy difference between different models. Ideally, we could use the model with highest accuracy in our system. 

## Existing Problem(s)
For some mac users, when using this command,
```
python webcam_detection.py --dataset dir_name
```
there might exists problem on terminal:
```
NotADirectoryError: [Errno 20] Not a directory: '/.DS_Store'
```
You can simply remove this file `.DS_Store` under testing directory.
We are still working on this, but other usages can be completed without any problem.

## Future
Here are the thoughts:

* We might add multiple faces detection later;
* Improve the accuracy using different Neural network model.

## Contact
Feel free to contact us.







