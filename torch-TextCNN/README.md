# Convolutional Neural Networks (CNN) for Text Classification

### 1. Overview
Text classification is a fundamental task in Natural Language Processing (NLP), involving the assignment of predefined categories to textual data. It has a wide range of applications, including sentiment analysis, spam detection, topic labeling, and intent recognition. Traditional approaches to text classification often relied on hand-crafted features and shallow models, which required domain expertise and struggled to generalize across different datasets.

In recent years, deep learning models have significantly improved performance in text classification tasks by automatically learning hierarchical representations from raw text. Among these models, Convolutional Neural Networks (CNNs) have emerged as a powerful and efficient architecture for capturing local patterns in text, such as n-grams, without relying on sequential processing. Originally designed for image recognition, CNNs have been successfully adapted to textual data by treating a sentence or document as a one-dimensional sequence of word embeddings. Through the application of convolutional filters of various sizes, CNNs can extract features that represent meaningful phrases or word combinations. These features are then aggregated through pooling operations and passed to fully connected layers for classification.

Compared to Recurrent Neural Networks (RNNs), CNNs offer advantages in parallelization, computational efficiency, and capturing local context. Their ability to detect discriminative phrases regardless of position makes them particularly suitable for tasks like sentiment classification, where certain words or expressions are strong indicators of class labels. This project leverages CNNs to perform binary sentiment classification on the IMDB movie review dataset, demonstrating the effectiveness of convolutional architectures in text classification scenarios.


### 2. Prepare .csv file

Before running the code, you need to make sure your .csv file follows the fomat with 2 column only. One for text and the other for label, the structure is depicted as follows. (you can also checked on the format of *IMDB_Dataset.csv*. The data under label coulmn could be numericalized or not.

|text|label|
|---|---|
|I am happy|positive|
|I hate you|negtive|


### 3. Run the code

There are two ways that enable to train on your dataset.

#### With Jupyter notebook

*textCNN_IMDB.ipynb* contains a complete procedure of sentiment analysis on IMDB dataset as provided in this repo, which will let you quickly train (Simply change the path in first line of code) with some pre-defined parameters. The version with argparser will be updated soon.

```
df = pd.read_csv('your file path')
```

#### With main.py

Directly execute the main.py with prefered parameters, the details of parameters is provided in following section.

```
python main.py --data-csv ./IMDB_Dataset.csv --spacy-lang en --pretrained glove.6B.300d --epochs 10 --lr 0.01 --batch-size 64 
--val-batch-size 64 --kernel-height 3,4,5 --out-channel 100 --dropout 0.5 -num-class 2
```

#### Parameters
```
  -h, --help                         show this help message and exit
  --data-csv DATA_CSV                file path of training data in CSV format (default:./train.csv)
  --spacy-lang SPACY_LANG            language choice for spacy to tokenize the text (default:en)
  --pretrained PRETRAINED            choice of pretrined word embedding from torchtext (default:glove.6B.300d)
  --epochs EPOCHS                    number of epochs to train (default: 10)
  --lr LR                            learning rate (default: 0.01)
  --momentum MOMENTUM                SGD momentum (default: 0.9)
  --batch-size BATCH_SIZE            input batch size for training (default: 64)
  --val-batch-size VAL_BATCH_SIZE    input batch size for testing (default: 64)
  --kernel-height KERNEL_HEIGHT      kernels for convolution (default: 3, 4, 5)
  --out-channel OUT_CHANNEL          output channel for convolutionaly layer (default: 100)
  --dropout DROPOUT                  dropout rate for linear layer (default: 0.5)
  --num-class NUM_CLASS              number of category to classify (default: 2)
```
**Pleace check out the available option for '--spacy-lang' at [spacy language support](https://spacy.io/usage/models#languages) 
and '--pretrained' at [torchtext pretrained_alias](https://torchtext.readthedocs.io/en/latest/vocab.html#pretrained-aliases)**

### 4. Check the results
The *main.py* will produce two simple plot that helps you check if there is any probelm with traning. Moreover, the best validation parameters will be stored in the same directory. 

## Reference
* https://stackoverflow.com/questions/43697240/how-can-i-split-a-dataset-from-a-csv-file-for-training-and-testing
* http://anie.me/On-Torchtext/
* https://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/
* https://torchtext.readthedocs.io/en/latest/examples.html
* https://github.com/pytorch/text
* https://github.com/pytorch/examples/blob/master/mnist/main.py
* https://github.com/Shawn1993/cnn-text-classification-pytorch
