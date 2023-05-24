# Facial-Emotion-Recognition-Project
A project about recognizing an emotion based on observed facial features using the FER 2013 Dataset.
The emotions detected in this project are: Happiness, Anger, Fear, Sadness, Disgust, and Neurality.

This project was made using python's library, Tensorflow.

![image](https://github.com/MohamadMulham/Facial-Emotion-Recognition-Project/assets/113246903/82ab103b-6124-4d5c-bc1a-91c8b6176e9c)

![image](https://github.com/MohamadMulham/Facial-Emotion-Recognition-Project/assets/113246903/c3879ef4-536e-41c7-b36f-ea92e052ef7c)

![image](https://github.com/MohamadMulham/Facial-Emotion-Recognition-Project/assets/113246903/fe65634f-5eb5-42e4-aa5a-cae9449d666b)

![image](https://github.com/MohamadMulham/Facial-Emotion-Recognition-Project/assets/113246903/c8b3c13b-1512-4bb6-8401-26546d8fdbf3)


**Instruction Manual:**
If you want to train the model yourself:
1-Download the FER2013 dataset and merge the testing and training files.
2-Specify the parameters like shape, epochs, paths to dataset and such.
3-Run the file and wait for it to finish executing.


**If you want the demo:**
1-Put the testing file in the same directory of the model structure(json file) and the weights(h5 file) and run the python file.





**Important notes:**
1-I used 6 of the 7 emotions from the dataset due to the imbalance between the number of examples
![image](https://github.com/MohamadMulham/Facial-Emotion-Recognition-Project/assets/113246903/56adc220-1124-408e-afd0-89024834c832)

2-The project uses a merged dataset where we split the training, validation, and testing splits ourselves.

3-The project is made using a 80, 10, 10 split as it is the standard.

4-The project is done using the adam optimizaer function and sparse categorical cross-entropy for a loss function as they are the current best optimizers.

5-The project experimented on 2 pretrained CNNs and 1 Custom CNN and the VGG19 was the highest in accuracy.


The training file of this project has multiple sections:
1-Importing necessary libraries.
2-Loading the merged dataset images and opening them.
3-Shuffling the image array order.
4-Spliting the dataset to training, validation, and testing splits.
5-Loading a pretrained VGG19 CNN and freezing all layers.
6-Using some great hyperparameters for the adam function, we specified the optimization function
7-Adding a couple of layers infront of the frozen CNN and the last layer with the number of classes we want to identify.
8-Adding a checkpoint that monitors the validation loss values and decides whether to save the last model or not based on the minimum value of loss.
9-Train the model.
10-Plot the loss and accuracy plots for training and validation.
11-Save the model and load the best weights.
12-Predict the classes of the testing split and form the Confusion Matrix.





**Structures and results:**

**VGG19**
Structure:

![vgg](https://github.com/MohamadMulham/Facial-Emotion-Recognition-Project/assets/113246903/61c7cf22-4649-4a4d-a17a-b4b095494587)


Figures:

![Accuracy Graph_small](https://github.com/MohamadMulham/Facial-Emotion-Recognition-Project/assets/113246903/0463a647-7c3c-4ea2-957c-fcceae557c68)

![Loss Graph_small](https://github.com/MohamadMulham/Facial-Emotion-Recognition-Project/assets/113246903/b7577b99-97bb-424a-8125-bcaa7b25bab3)

Confusion Matrix:

![Confusion Matrix](https://github.com/MohamadMulham/Facial-Emotion-Recognition-Project/assets/113246903/183f270f-4f10-438c-907a-5c43012878f5)



**Custom Structure**

Structure:

![image](https://github.com/MohamadMulham/Facial-Emotion-Recognition-Project/assets/113246903/d7f3a914-26de-4b43-a00d-d2cca6b7578d)


Figures:

![Accuracy Graph_small](https://github.com/MohamadMulham/Facial-Emotion-Recognition-Project/assets/113246903/a1ff9fc7-1016-4531-820e-bcbfa8ec40af)

![Loss Graph_small](https://github.com/MohamadMulham/Facial-Emotion-Recognition-Project/assets/113246903/1d5df7f4-6934-4e34-8427-89c399859af9)


Confusion Matrix:

![Confusion Matrix](https://github.com/MohamadMulham/Facial-Emotion-Recognition-Project/assets/113246903/b5348691-6997-4b14-9029-7129c91657b3)



**MobileNet 2**
Structure:

![mobile](https://github.com/MohamadMulham/Facial-Emotion-Recognition-Project/assets/113246903/bef10353-3056-49b4-b3b1-87b64e77de7c)


Figures:

![Accuracy Graph_small](https://github.com/MohamadMulham/Facial-Emotion-Recognition-Project/assets/113246903/ac8e4d8f-4420-481c-bcf8-6bf1b8595ab6)

![Loss Graph_small](https://github.com/MohamadMulham/Facial-Emotion-Recognition-Project/assets/113246903/18ad39dc-2823-4e45-961c-e3821319a201)


Confusion Matrix:

![Confusion Matrix](https://github.com/MohamadMulham/Facial-Emotion-Recognition-Project/assets/113246903/d722cb9f-56e1-4a0e-8303-a3b94c9fe94c)











Keywords:
Machine Learning, Supervised Learning, CNN, Deep Learning, Pretrained CNN, Confusion Matrix, MobileNet, VGG, VGG19, Emotion, Emotion Recognition, Facial Emotion, Facial Emotion Recognition, Demo, Training, Testing, Validation.




