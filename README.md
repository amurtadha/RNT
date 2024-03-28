 
 # RNT 
 A model  for learning under semi-supervised settings
 
 This is the source code for the paper: Murtadha, Ahmed, et al. "Rank-Aware Negative Training for Semi-Supervised Text Classification". 

# Data



The datasets used in our experminents can be downloaded from this [link](https://drive.google.com/file/d/1NYm5CVXK7vqn-Nf8rnin-4iAxWeJcKVv/view?usp=sharing). 

# Prerequisites:
Required packages are listed in the requirements.txt file:

```
pip install -r requirements.txt
```
# Training

*  Go to code/         
*  Run the following code to train RNT:
```
python run.py --dataset='SST-5' --train-sample=30
```

- The params could be :
    - --dataset =\{AG,yelp, yahoo, TREC,SST, SST-5, CR, MR\}
    - --train_sample ={0, 30,1000, 10000}, where 0 denotes 10% of the labeled data

The results will be written to results/main_nt.txt

# Evaluation

*  Go to code/         
*  Run the following code to evaluate RNT:
```
python evaluate.py --dataset='SST-5' --train-sample=30
```


If you use the code,  please cite the paper: 
 ```
@article{RNT-TACL-2023,
  author       = {Ahmed Murtadha and
                  Shengfeng Pan and
                  Wen Bo and
                  Jianlin Su and
                  Xinxin Cao and
                  Wenze Zhang and
                  Yunfeng Liu},
  title        = {Rank-Aware Negative Training for Semi-Supervised Text Classification},
  journal      = {Transactions of the Association for Computational Linguistics (TACL, 2023)},
  volume       = {abs/2306.07621},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2306.07621},
  doi          = {10.48550/arXiv.2306.07621}
}
```
 
