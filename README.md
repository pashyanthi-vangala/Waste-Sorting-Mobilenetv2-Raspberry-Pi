# Low-Cost Automated Waste Sorting System Using MobileNetV2 and Raspberry Pi

A low-cost waste sorting system that uses computer vision, MobileNetV2, and a Raspberry Pi to classify different types of waste and automatically sort them into the appropriate bin.

## Overview

This project was developed as part of UCSD CSE SPIS 2026.

The system uses a camera connected to a Raspberry Pi to take a picture of a waste item. A trained MobileNetV2 model classifies the item into one of six categories:

* Cardboard
* Glass
* Metal
* Paper
* Plastic
* Trash

Although the model has six different classes, the physical system has **two bins: recyclable and trash**. We used six classes because separating the different types of waste helps the model make a more specific prediction before deciding which of the two bins the item should go into.

The system also uses a confidence threshold. If the model is not confident enough about its prediction, the item is automatically sent to the trash bin instead of making a potentially incorrect sorting decision.

## How It Works

1. A waste item is placed in front of the camera.
2. A button connected to the Raspberry Pi starts the classification process.
3. The camera captures an image of the item.
4. The image is resized to `128 × 128` pixels.
5. The MobileNetV2 model predicts one of the six waste categories.
6. The model's confidence is checked against a **60% confidence threshold**.
7. If the confidence is below 60%, the item is sent to the trash bin.
8. If the prediction is confident enough, the system determines whether the predicted class belongs in the recyclable or trash bin.
9. A servo motor moves the sorting mechanism toward the appropriate bin.
10. The servo returns to its neutral position after sorting.

The six-class classification gives the model more specific information about the type of waste, while the physical system simplifies the final decision into two categories.

## Machine Learning

The model uses **MobileNetV2** with ImageNet pretrained weights.

We first experimented with different CNN architectures and training techniques as a baseline. After testing different approaches, we used MobileNetV2 and fine-tuned the pretrained model for the waste classification task.

The final model classifies six types of waste:

| Class     | Description                  |
| --------- | ---------------------------- |
| Cardboard | Cardboard waste              |
| Glass     | Glass waste                  |
| Metal     | Metal waste                  |
| Paper     | Paper waste                  |
| Plastic   | Plastic waste                |
| Trash     | General non-recyclable waste |

### Why Six Classes?

The physical system only has two bins, but we chose to classify the waste into six separate classes first. Materials such as cardboard, paper, plastic, glass, and metal can have different appearances and can sometimes be difficult to distinguish from one another, even for people.

Using separate classes allows the model to learn the visual differences between these materials instead of treating all recyclable items as one large category. The individual predictions are then mapped to the two physical bins.

### Confidence Threshold

The system uses a **60% confidence threshold**.

When the model's highest prediction is below this threshold, the system does not rely on the uncertain prediction and sends the item to the trash bin.

This was added because an incorrect sorting decision could be more useful to avoid than forcing the system to make a low-confidence classification.

### Model Configuration

* Model: MobileNetV2
* Input size: `128 × 128 × 3`
* Number of classes: 6
* Optimizer: Adam
* Learning rate: `1e-5`
* Dropout: `0.3`
* Early stopping: Used during training
* Framework: TensorFlow / Keras

## Results

The final model achieved approximately **88.33% accuracy on the test dataset**.

The model was evaluated using:

* Test accuracy
* Confusion matrix
* Classification report

More details about the experiments and model comparisons will be added to this repository.

## Hardware

The physical sorting system uses:

* Raspberry Pi
* Camera
* Push button
* Servo motor
* Two waste bins
* Sorting mechanism

The Raspberry Pi runs the trained model and controls the camera, button, and servo motor.

## Software

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* libgpiod
* lgpio
* Raspberry Pi camera software

## Dataset

The waste images used for training, validation, and testing were obtained from the **TrashBusters Combined** dataset on Hugging Face.(https://huggingface.co/datasets/TrashBusters/combined)

The dataset was organized into training, validation, and test sets and contains images from the six waste categories used by the model.

## Project Images

The `images` folder contains photos of the Raspberry Pi and the physical waste sorting system.

## Experiments and Results

Detailed documentation of the CNN experiments, transfer learning, MobileNetV2 fine-tuning, final evaluation, and Raspberry Pi deployment is available in the Experiments and Results document.

## Future Improvements

Some possible improvements for the system include:

* Improving classification accuracy with a larger and more diverse dataset
* Adding more types of waste and additional bins
* Improving the physical sorting mechanism
* Making the system more compact and portable
* Testing the system under different lighting and background conditions
* Improving the handling of low-confidence predictions
  
## Project

This project was created as a pair project during **UCSD CSE SPIS 2026** by:

* **Pashyanthi Vangala**
* **Sherry Zhang**

