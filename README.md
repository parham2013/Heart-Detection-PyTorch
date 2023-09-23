# What is Heart Disease?
Heart disease is the leading cause of death in the United States, the UK, and worldwide. It causes more than 73,000 and 600,000 deaths per year in the UK and the US, respectively[1](https://www.nature.com/articles/s41598-023-34294-6#ref-CR1),[2](https://www.nature.com/articles/s41598-023-34294-6#ref-CR2). Heart disease caused the death of about 1 in 6 men and 1 in 10 women.  
Source: [Nature Journal](https://www.nature.com/articles/s41598-023-34294-6)


![image](https://github.com/parham2013/Heart-Detection-PyTorch/assets/74326920/1f0fec6d-9cd1-4714-ad47-c6149dbd67fa)  
Image Source: [Wikipedia](https://en.wikipedia.org/wiki/Coronary_artery_disease)


# How Does Machine Learning Help?  
Machine learning analyzes patient data to identify heart disease patterns. It assists in accurate diagnoses and predicts patient outcomes. Wearable devices can integrate these algorithms for real-time heart health monitoring.








# Goal of This Project
The goal is to train a machine learning model that predicts a box around the heart in X-ray images.  

Some of the predicted boxes:  

![image](https://github.com/parham2013/Heart-Detection-PyTorch/assets/74326920/95f89d41-cec9-447e-89f1-5c738c3db821)


# How is This Helpful?

Anomalies in the size or position of the heart can help us predict heart disease


Project consists of 4 sections:  
- Preprocessing
- Dataset
- Model-Training
- Interpretability

## Preprocessing
Separate normalize and separate our data into train-val, and also calculate mean and std of pixel arrays and save their labels for later.  

Few images from training data:  

![image](https://github.com/parham2013/Heart-Detection-PyTorch/assets/74326920/3432b144-5d96-4322-a0f7-41ad87910ffa)

## Dataset
Create a CardiacDataset class to extract bounding box coordinates and augment images with boxes together, if there was augmentation available,
also normalize images with mean and std saved.
We save this class as a cardiac_dataset.py script to import it in Model-Training later.  

Example of image and box, both augmented in the same way:  

![image](https://github.com/parham2013/Heart-Detection-PyTorch/assets/74326920/e43869e7-ce63-46ee-a957-45f6cbcdfe55)

## Model-Training

Loading datasets with corresponding paths to labels and actual data, defining data loader and then  
creating CardiacDetectionModel, we used pretrained ResNet18 model with minor modification, using optim.Adam and MSELoss,
we also logged images for better visualization of how actually the training happens:  

![output](https://github.com/parham2013/Heart-Detection-PyTorch/assets/74326920/0666c89e-dacc-4a6b-81ff-59c46ac852cc)

## Interpretability
Using the model is very simple, do the following:
1. Download the Trained-Model from releases
2. Download interpretability notebook and put it in the same folder as Trained-Model
3. Create a Python venv, install libraries
4. change the image address  with the address of your own image(Images must be in Dicom format)

