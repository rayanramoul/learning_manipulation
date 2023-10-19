# Learning Manipulation
This repository is related on studying methodologies to orient the feature space used by model to train.

## How to install ?
```
git clone git@github.com:rayanramoul/learning_manipulation.git
conda env create -f environment.yml
conda activate learning
python main.py
```


## Get the dataset 
Connect to Kaggle and get the Dogs vs Cats dataset from this link:
https://www.kaggle.com/competitions/dogs-vs-cats/data

## How to use ?

## Next Steps
- [X] Add feature maps visualization.
- [X] Genearalize to multiple images the visualization.
- [X] Verify calculus of gradcam.
- [X] Think on how to verbalize the non-explainability of the features used.
- Test multi  cut training combined with other metrics.
-  Split feature maps visualization from gradcam.
- At some point prune the network to use in the convolution part only the kernels or pixels that maximize the information (think about edge cases like some small set of images which sometime activate pixels/kernels that otherwise are never activated)