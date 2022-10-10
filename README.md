 
 # RNT 
 A model  for learning under semi-supervised settings
 
 This is the source code for the paper: Murtadha, Ahmed, et al. "Rank-Aware Negative Training for Semi-Supervised Text Classification". If you use the code,  please cite the paper: 
 ```
```
 

# Data



The datasets used in our experminents can be downloaded from this [link](https://drive.google.com/file/d/1NYm5CVXK7vqn-Nf8rnin-4iAxWeJcKVv/view?usp=sharing). 

# Prerequisites:
Required packages are listed in the requirements.txt file:

```
pip install -r requirements.txt
```
# How to use

*  Go to code/         
*  Run the following code to train under noisy label settuings:
```
python run.py --dataset='SST-5' --train-sample=30
```

- The params could be :
    - --dataset =\{AG,yelp, yahoo, TREC,SST, SST-5, CR, MR\}
    - --train_sample ={0, 30,1000, 10000}, where 0 denotes 10% of the labeled data

The results will be written to results/main_nt.txt

