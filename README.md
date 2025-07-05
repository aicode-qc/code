# Embodied Cognition-Driven Interpretable Trajectory Prediction of Autonomous System

## 1. Environment Requirements
The code was written using Python 3.9. The minimal libraries required to run the code are:
```python
import pytorch
import networkx
import numpy
import tqdm
```
or you can have everything set up by running:
、、、
pip install -r requirements.txt
、、、
## 2. Model Training
To train a model for each dataset with the best configuration as described in the paper, execute:
```python
./train.sh
```

## 3. Model Evaluation
To utilize the pretrained models located at checkpoint/ and evaluate their performance, run:
```python
python test.py
```
