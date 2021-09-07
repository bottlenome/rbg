# Random Batch Generalization
Pytorch implementation of our paper.
Satoru Mizusawa and Yuichi Sei: Interlayer Augmentation in a Classification Task, 4th IEEE International Conference on Computing, Electronics & Communications Engineering (iCCECE), pp.59-64 (2021.8)

# Install
pip install git+https://github.com/bottlenome/rbg.git

# Sample
see train.py

# Simple Usage
1. import RBG or BG
```python
from rbg.generalization import RandomBatchGeneralization, BatchGeneralization 

```
2. Apply to model
See rgb/model.py WrapVGG
