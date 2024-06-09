# Real-Time Face Detection, Gender, and Age Prediction

This repository contains a Python script that performs real-time face detection, gender prediction, and age estimation using a webcam feed. The script leverages deep learning models to analyze video frames, detect faces, and predict the age and gender of each detected face. This project is designed for applications in various fields such as security, marketing analytics, and user experience enhancement.

## Features

- **Real-Time Face Detection**: Uses a pre-trained face detection model to locate faces in each frame of the video feed.
- **Gender Prediction**: Predicts the gender (male or female) of each detected face using a deep learning model.
- **Age Estimation**: Estimates the age range of each detected face using a deep learning model.
- **Webcam Integration**: Captures video from the webcam and processes it in real-time.
- **Visualization**: Displays the video feed with bounding boxes around detected faces along with their name, age and gender.

## Requirements

- Python 3.6+
- OpenCV
- imutils
- scikit-learn
- pickel
- cmake
- dlib (Download from [here](https://github.com/z-mahmud22/Dlib_Windows_Python3.x/tree/main))
- Pre-trained models for face detection, age prediction, and gender prediction

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/real-time-face-gender-age-detection.git
    cd real-time-face-gender-age-detection
    ```

2. Additionally, install the following packages:
    ```bash
    pip install imutils
    pip install pickel
    pip install scikit-learn
    pip install cmake
    ```

3. Install dlib. For Windows users, follow the instructions and download dlib from [this repository](https://github.com/z-mahmud22/Dlib_Windows_Python3.x/tree/main).


## Usage

### Embedding New Data

To train the model with new data, add a folder with the person's name and add photos to the relevant folder. Then run the following command to extract embeddings:

```bash
python extract_embeddings.py --dataset dataset --embeddings output/embeddings.pickle --detector face_detection_model --embedding-model nn4.small2.v1.t7
```

### Training the Model

After extracting embeddings, train the model using:

```bash
python train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle
```

### Running the Real-Time Recognition

Finally, run the live recognition script:

```bash
python live_recognition.py --detector face_detection_model --embedding-model nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle
```

## File Structure

- `extract_embeddings.py`: Script to extract facial embeddings from the dataset.
- `train_model.py`: Script to train the recognition model.
- `live_recognition.py`: Script to run the real-time face detection, gender, and age prediction.
- `models/`: Directory to store the pre-trained models.
- `output/`: Directory to store the output embeddings and trained models.

## Pre-trained Models

The script requires pre-trained models for face detection, gender prediction, and age estimation. These Models have provided but you can replace them with your own to improve accuracy!


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [OpenCV](https://opencv.org/)
- [scikit-learn](https://scikit-learn.org/)
- [imutils](https://github.com/jrosebr1/imutils)

## Contact

For any questions or suggestions, please open an issue or contact me at mrafaym015@gmail.com