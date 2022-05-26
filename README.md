# Sinkhorn-Network

> [introduction blog](https://zhuanlan.zhihu.com/p/76742946)

This is a simple Sinkhorn Network model implementation with Pytorch refer to [paper](https://openreview.net/pdf?id=Byt3oJ-0W).  Here is the official [tensorflow implementation](https://github.com/google/gumbel_sinkhorn).

Our Task
* Sorting the object permutations.


![img](https://pic2.zhimg.com/80/v2-ce93fdbebae3c4991de6303f9143d355_720w.jpg)

![img](https://pic2.zhimg.com/80/v2-b647a176a42547659da525898e00e511_720w.jpg)



model file: 
  * [object_class_glove.pkl](https://drive.google.com/file/d/1na9xQ7PJvn2nYj2KpH9C6pEAlYWmJnPt/view?usp=sharing)
  * genome-trainval.h5 (on my personal server)

dataset: 
* [folder: ./data/datasets/sentence_pairs.json](./data/datasets/sentence_pairs.json)

## Data Preprocess
```angular2html
cd ./data
python prepor.py
```
After this step, train, validation and test data will generate under the `./data/datasets` folder.

## Main Process

Train and Validation
```python
python train.py
```
Test
```python
python test.py
```


