# Easy_Tensor
Unzip ```open.zip``` file which you can download from dacon competition page into ```./data``` directory.

- Hierarchy

```
2024SWAICompetition
+-- data
|   +-- open
|   |   +-- test
|   |   +-- test_emb
|   |   +-- train
|   |   +-- train_aug
|   |   |   +-- 0
|   |   |   +-- 1
|   |   |   +-- 2
|   |   |   +-- 3
|   |   |   +-- 4
|   |   |   +-- 5
|   |   +-- train_aug_emb
|   |   |   +-- 0
|   |   |   +-- 1
|   |   |   +-- 2
|   |   |   +-- 3
|   |   |   +-- 4
|   |   |   +-- 5
|   |   +-- train_emb
|   |   +-- unlabeled_data
+-- ensemble
+-- history
+-- ...
```

<hr/>

## Description

- ```./data/```: Data Directory<br/>
- ```./ensemble/```: Parameters for Ensembled Inference<br/>
- ```./history/```: Recorded Histories While Training<br/>
- ```./src/```: Source Files for Markdown<br/>
- ```./submission/```: Results of Inference Session<br/>
- ```./WeSpeaker_ResNet221/```: Embedding Network Directory<br/>
- ```./audio_to_embedding.py```: Convert Every Audio Files to Embedding Vectors<br/>
- ```./inference_multi_ensemble.py```: Make a New Inference File or Reproduce Previous Inference File<br/>
- ```./main.py```: Train Network with New Parameter<br/>
- ```./MainDataset.py```: Custom Dataset for Actual Training<br/>
- ```./MixedAudioDataset.py```: Data Augmentation Session Before Training<br/>
- ```./model.py```: Our Cosine-Similarity-Based Siamese Network<br/>
- ```./pred_analysis.py```: Check the Distribution of Prediction for Submission<br/>
- ```./test.py```: Calculate Each Metric Score of Every Parameter File<br/>
- ```./utils.py```: Utility Functions for this Project<br/>
- ```./config.yaml```: Configurations and Hyperparameters for Training
![alt text](https://github.com/kylekim00/Easy_Tensor/blob/main/CAPSTONE/1.JPG?raw=true)
![alt text](https://github.com/kylekim00/Easy_Tensor/blob/main/CAPSTONE/2.JPG?raw=true)
![alt text](https://github.com/kylekim00/Easy_Tensor/blob/main/CAPSTONE/3.JPG?raw=true)
![alt text](https://github.com/kylekim00/Easy_Tensor/blob/main/CAPSTONE/4.JPG?raw=true)
![alt text](https://github.com/kylekim00/Easy_Tensor/blob/main/CAPSTONE/5.JPG?raw=true)
![alt text](https://github.com/kylekim00/Easy_Tensor/blob/main/CAPSTONE/6.JPG?raw=true)
![alt text](https://github.com/kylekim00/Easy_Tensor/blob/main/CAPSTONE/7.JPG?raw=true)
![alt text](https://github.com/kylekim00/Easy_Tensor/blob/main/CAPSTONE/8.JPG?raw=true)
![alt text](https://github.com/kylekim00/Easy_Tensor/blob/main/CAPSTONE/9.JPG?raw=true)
![alt text](https://github.com/kylekim00/Easy_Tensor/blob/main/CAPSTONE/10.JPG?raw=true)
![alt text](https://github.com/kylekim00/Easy_Tensor/blob/main/CAPSTONE/10.JPG?raw=true)
![alt text](https://github.com/kylekim00/Easy_Tensor/blob/main/CAPSTONE/11.JPG?raw=true)
![alt text](https://github.com/kylekim00/Easy_Tensor/blob/main/CAPSTONE/12.JPG?raw=true)
![alt text](https://github.com/kylekim00/Easy_Tensor/blob/main/CAPSTONE/13.JPG?raw=true)
![alt text](https://github.com/kylekim00/Easy_Tensor/blob/main/CAPSTONE/14.JPG?raw=true)
![alt text](https://github.com/kylekim00/Easy_Tensor/blob/main/CAPSTONE/15.JPG?raw=true)
![alt text](https://github.com/kylekim00/Easy_Tensor/blob/main/CAPSTONE/16.JPG?raw=true)
![alt text](https://github.com/kylekim00/Easy_Tensor/blob/main/CAPSTONE/17.JPG?raw=true)
![alt text](https://github.com/kylekim00/Easy_Tensor/blob/main/CAPSTONE/18.JPG?raw=true)
![alt text](https://github.com/kylekim00/Easy_Tensor/blob/main/CAPSTONE/19.JPG?raw=true)
![alt text](https://github.com/kylekim00/Easy_Tensor/blob/main/CAPSTONE/20.JPG?raw=true)
![alt text](https://github.com/kylekim00/Easy_Tensor/blob/main/CAPSTONE/21.JPG?raw=true)
![alt text](https://github.com/kylekim00/Easy_Tensor/blob/main/CAPSTONE/22.JPG?raw=true)
![alt text](https://github.com/kylekim00/Easy_Tensor/blob/main/CAPSTONE/23.JPG?raw=true)
![alt text](https://github.com/kylekim00/Easy_Tensor/blob/main/CAPSTONE/24.JPG?raw=true)
![alt text](https://github.com/kylekim00/Easy_Tensor/blob/main/CAPSTONE/25.JPG?raw=true)
![alt text](https://github.com/kylekim00/Easy_Tensor/blob/main/CAPSTONE/26.JPG?raw=true)
![alt text](https://github.com/kylekim00/Easy_Tensor/blob/main/CAPSTONE/27.JPG?raw=true)
![alt text](https://github.com/kylekim00/Easy_Tensor/blob/main/CAPSTONE/28.JPG?raw=true)
![alt text](https://github.com/kylekim00/Easy_Tensor/blob/main/CAPSTONE/29.JPG?raw=true)
![alt text](https://github.com/kylekim00/Easy_Tensor/blob/main/CAPSTONE/30.JPG?raw=true)
![alt text](https://github.com/kylekim00/Easy_Tensor/blob/main/CAPSTONE/31.JPG?raw=true)
![alt text](https://github.com/kylekim00/Easy_Tensor/blob/main/CAPSTONE/32.JPG?raw=true)
![alt text](https://github.com/kylekim00/Easy_Tensor/blob/main/CAPSTONE/33.JPG?raw=true)
![alt text](https://github.com/kylekim00/Easy_Tensor/blob/main/CAPSTONE/34.JPG?raw=true)
![alt text](https://github.com/kylekim00/Easy_Tensor/blob/main/CAPSTONE/35.JPG?raw=true)
![alt text](https://github.com/kylekim00/Easy_Tensor/blob/main/CAPSTONE/36.JPG?raw=true)
![alt text](https://github.com/kylekim00/Easy_Tensor/blob/main/CAPSTONE/37.JPG?raw=true)
![alt text](https://github.com/kylekim00/Easy_Tensor/blob/main/CAPSTONE/38.JPG?raw=true)
![alt text](https://github.com/kylekim00/Easy_Tensor/blob/main/CAPSTONE/39.JPG?raw=true)
![alt text](https://github.com/kylekim00/Easy_Tensor/blob/main/CAPSTONE/40.JPG?raw=true)
