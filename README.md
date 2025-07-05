# Embodied Cognition-Driven Interpretable Trajectory Prediction of Autonomous System

##  Environment Requirements
The code was written using Python 3.9. The minimal libraries required to run the code are:
```python
import pytorch
import networkx
import numpy
import tqdm
```
##  Environment Requirements
or you can have everything set up by running:
```python
pip install -r requirements.txt
```
##  Model Training
To train a model for each dataset with the best configuration as described in the paper, execute:
```python
./train.sh
```

## Model Evaluation
To utilize the pretrained models located at checkpoint/ and evaluate their performance, run:
```python
python test.py
```
