# Neural Network Models (nn-models)

Welcome to the `nn-models` repository! This repository is a collection of various neural network implementations, exploring different architectures and applications, from computer vision to generative music.

---

## Projects

This repository is organized into the following projects:

### 1. CNN Implementations (`/cnn-scratch`)

This project focuses on understanding and building Convolutional Neural Networks (CNNs) from the ground up. It includes multiple implementations and breaks down the core components of a CNN.

**Key Files & Features:**
* `cnn_scratch.py`: A full CNN model built from scratch using basic libraries.
* `cnn_keras.py`: A comparative implementation of the same CNN architecture using the Keras library.
* `conv.py`: Standalone implementation of the Convolutional layer.
* `maxpool.py`: Standalone implementation of the Max Pooling layer.
* `softmax.py`: Standalone implementation of the Softmax activation function.

### 2. Music Generation (`/music-generation`)

This project explores the use of neural networks to generate new musical pieces. It includes scripts for MIDI data manipulation, model training, and music generation.

**Key Files & Features:**
* `neuralnet.py`: The core neural network model used for music generation.
* `midi_manipulation.py`: Contains functions for processing and handling MIDI files.
* `main.py`: The main script to run the music generation pipeline.
* `miditomp3.py`: A utility script to convert MIDI files to MP3 format.
* `EDEN-midi/`: A directory likely containing the dataset of MIDI files used for training.
* `generated/`: The default output directory for generated music files.
* `output.mp3`: An example of a generated audio file.


## Technologies Used

* **Language:** Python
* **Libraries (Probable):**
    * **CNN:** TensorFlow, Keras, NumPy, OpenCV
    * **Music Generation:** mido, music21, pydub, NumPy, TensorFlow/PyTorch


## How to Use

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Miraj-Rahman-AI/nn-models.git](https://github.com/Miraj-Rahman-AI/nn-models.git)
    ```

2.  **Navigate to a project directory:**
    ```bash
    # For the CNN project
    cd nn-models/cnn-scratch

    # For the Music Generation project
    cd nn-models/music-generation
    ```

3.  **Install the required dependencies:**
    *(It is recommended to create a `requirements.txt` file in each project directory)*
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the main script for the project:**
    ```bash
    python main.py 
    # or
    python cnn_scratch.py
    ```



