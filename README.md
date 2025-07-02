![image](https://github.com/user-attachments/assets/52f59383-7707-4b0a-b7fa-8ea8dafac0df)
### Convolutional neural network for vibrational data analysis. 
This network was designed for vibrational data analysis, utilizing the capabilities of convolutional networks for feature extraction from more than one source (channels). It has been tested on large datasets, achieving experimental performance above 90% for four classes. This approach was developed for analyzing signals from two accelerometers mounted on a 1HP induction motor, using the dataset created by Case Western Reserve University, which is available on their [website](https://engineering.case.edu/bearingdatacenter/download-data-file). The dataset includes data representing a healthy state and three types of failures.

This repository contains three descriptions. The first one, data_management, is used for the Dataloader class. In this file, the data structure is preprocessed (as a single-channel vector) and converted into a matrix representation. Subsequently, two independent *1D* vectors (*256* samples each) are converted into a *2x16x16* tensor.

The second description, conv_model, contains the structure of the model, which was built using PyTorch libraries and invokes the Dataloader class. It also includes the training routine, which uses a stop criterion experimentally set to prevent overfitting during the training process. This routine outputs the model’s performance as a single percentage, representing the average accuracy across the four classes.

The third description, conv_model_metrics, contains the same structure of conv_model but some routines, graphics, and metrics were added to show loss graphic as well as confusion matrices for train and test datasets.

The data structure in the files 4C_general_007.csv, 4C_general_014.csv, and 4C_general_021.csv [available here](https://drive.google.com/drive/folders/1jBCRPD5igolbaiN9DhYOcXxsnnlgSDj2?usp=sharing), contains vibrational data corresponding to different severity levels (artificial holes with diameters of *0.007"*, *0.014"*, and *0.021"*). Each file consists of eight columns: the first four columns correspond to the end drive accelerometer signals for healthy, inner race failure, ball failure, and external race failure conditions, respectively. The remaining four columns follow the same order but are associated with the fan accelerometer signals.

It is highly recommended to store descriptions and files in the same directory.


The model performance can be evaluated using metrics as average accuracy for training and test datasets. However, this can also be visualized in confusion matrices to observe the data separability.

In Fig. 1 the test and train loss are displayed for 260 epochs. As can be observed, the optimization algorithm (SGD) converges gradually during the training process.

<p align="center">
  <img width="500" src="https://github.com/user-attachments/assets/6d78f611-ab7c-4811-b3f6-3e731de763b0">
</p>
<p align="center">
    <em>Fig. 1: Loss evolution vs epochs graphic, for training and test data.</em>
</p>

In Fig. 2 and Fig. 3 the test and train confusion matrix obtained from the model after 260-epoch training process are shown. The performance in this case was 99% of accuracy for both test and training data.

<p align="center">
  <img width="550" src="https://github.com/user-attachments/assets/639a073f-3f1e-473c-83d1-86b122dcf9d2">
</p>
<p align="center">
    <em>Fig. 2: Confussion matrix obtained after training for test data.</em>
</p>

<p align="center">
  <img width="550" src="https://github.com/user-attachments/assets/eaba9ef8-c9a3-4c5e-b8db-bdfba4064fa6">
</p>
<p align="center">
    <em>Fig. 3: Confussion matrix obtained after training for train data.</em>
</p>
