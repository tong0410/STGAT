# README

To run this code, you should first prepare some of the datasets. Different numpy files represent different features. And you should prepare nodes features and edges features separately. 

Our research provides a new perspective and a powerful tool for predicting the trajectory of biomacromolecules, and realizes the real-time dynamic monitoring of the trajectory of biomacromolecules.  And our model adeptly assimilates structural information from each protein’s molecular space, enabling accurate trajectory predictions. In practical applications, a synergistic integration of deep learning methods and traditional molecular dynamics simulations can offer a more comprehensive and precise molecular prediction and analysis framework.



The model begins by receiving various input features, which are then fed into an STGAT layer. This layer utilizes a sandwich structure, consisting of Time Convolution Layers and a GAT layer, to efficiently extract and integrate both temporal and graph-based information from the input features.

![pic1](https://github.com/tong0410/STGAT/tree/main/figure/pic1.png)



## Example(RS1)

* If you want to train the STGAT model on RS1 protein:

```bash
python ./main.py --dataset RS1 --epochs 300 
```

* After training the model on RS1, you should use the model to finish the prediction:

```bash
python ./tools/predict.py
```



