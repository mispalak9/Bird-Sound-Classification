<<<<<<< HEAD
# Bird Sound Classification using Deep Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16gteAoj-kv8GlMo4HMrovzpkqu9dS9yG?usp=sharing)


**Introduction**

In environmental conservation and wildlife research, identifying bird species by their vocalizations is crucial but challenging due to the diversity of calls and complex acoustic environments. Our project, Bird Sound Classification using Deep Learning, addresses this by using TensorFlow and Convolutional Neural Networks (CNNs) to accurately classify bird species from audio signals. This approach aims to enhance avian research, support ecological monitoring, and contribute to global bird conservation efforts.

<br />

**Table of Contents**

1. Key Technologies and Skills
2. Installation
3. Usage
4. Features
5. Contributing
6. License
7. Contact

<br />

**Key Technologies and Skills**
- Python
- TensorFlow
- Convolutional Neural Network (CNN)
- Keras
- scikit-learn
- OpenCV
- Numpy
- Pandas
- Matplotlib
- Streamlit
- Hugging Face

<br />

**Installation**

To run this project, you need to install the following packages:

```python
pip install tensorflow
pip install scikit-learn
pip install opencv-python
pip install librosa
pip install ipython
pip install numpy
pip install pandas
pip install matplotlib
pip install streamlit
pip install streamlit_extras
pip install tqdm
```

**Note:** If you face "ImportError: DLL load failed" error while installing TensorFlow,
```python
pip uninstall tensorflow
pip install tensorflow==2.12.0 --upgrade
```

<br />

**Usage**

To use this project, follow these steps:

1. Clone the repository: ```git clone https://github.com/mispalak9/Bird-Sound-Classification.git```
2. Install the required packages: ```pip install -r requirements.txt```
3. Run the Streamlit app: ```streamlit run app.py```
4. Access the app in your browser at ```http://localhost:8501```
5. Change Path: ```cd Bird-Sound-Classification-using-Deep-Learning```
6. Change Path: ```.\venv\Scripts\activate```
<<<<<<< HEAD

=======
>>>>>>> 0496fa2789c849142e33c2f428867c54a13291f1

<br />

**Features**

#### Data Collection:
   - The bird sound dataset utilized in this project was sourced from Kaggle, a prominent platform for datasets and data science resources. This dataset comprises 2161 audio files (mp3) capturing the vocalizations of 114 distinct bird species.

   - Each audio file is meticulously annotated with the corresponding bird species, providing crucial labels for supervised learning tasks in deep learning-based classification models.

📙 Dataset Link: [https://www.kaggle.com/datasets/soumendraprasad/sound-of-114-species-of-birds-till-2022](https://www.kaggle.com/datasets/soumendraprasad/sound-of-114-species-of-birds-till-2022)


#### Preprocessing:

   - **Audio Feature Extraction:** The preprocessing phase commences with the extraction of audio features from the raw audio files using the Librosa library. Specifically, we utilize Librosa to extract Mel Frequency Cepstral Coefficients (MFCC) from each audio signal, capturing crucial spectral characteristics essential for bird sound classification.

   - **TensorFlow Dataset Creation:** Following feature extraction, we convert the extracted MFCC features and corresponding target labels into TensorFlow tensors. These tensors are then encapsulated within a TensorFlow dataset, facilitating seamless integration into the deep learning model architecture.

   - **Data Splitting:** To ensure robust model evaluation, the dataset is partitioned into three subsets: training, validation, and testing. This partitioning scheme enables independent assessment of model performance during training, validation, and final testing stages, thereby enhancing model generalization and mitigating the risk of overfitting.

   - **Data Pipeline Optimization:** A key focus of preprocessing is the optimization of the data pipeline to enhance training efficiency. Leveraging TensorFlow's pipeline optimization techniques, such as `caching, shuffling, and prefetching`, we accelerate the data ingestion process and minimize training time. By proactively prefetching data batches and caching preprocessed samples, we mitigate potential bottlenecks and maximize GPU utilization, culminating in expedited model convergence and improved computational efficiency.


#### Model Building and Training:

   - **Model Architecture:** The model architecture is meticulously crafted using TensorFlow's Keras API. A Convolutional Neural Network (CNN) is constructed, featuring convolutional layers for feature extraction, pooling layers for spatial downsampling, and dense layers for classification. Adjustable filters, units, and activation functions are incorporated to tailor the model's capacity to the complexity of the dataset.

   - **Training:** Model training is orchestrated using an end-to-end pipeline encompassing data loading, preprocessing, model instantiation, and optimization. Leveraging the `Adam` optimizer, `sparse_categorical_crossentropy` loss function, and `Accuracy` metrics, we optimize the model parameters to minimize classification error. Throughout training, the model's performance is monitored on a separate validation dataset after each epoch to prevent overfitting and ensure generalization. Upon completion of training, the model attains a remarkable accuracy of **93.4%**, underscoring its proficiency in accurately classifying bird sounds.


#### Model Deployment and Inference:

   - Following the completion of model training and evaluation, the trained model is saved to enable seamless deployment and inference on new audio for classification purposes. To facilitate this process, a user-friendly Streamlit application is developed and deployed on the Hugging Face platform.

   - This application empowers users to upload new audio files and obtain real-time classification results, providing a convenient interface for leveraging the model's capabilities in practical scenarios.


<br />

<<<<<<< HEAD
=======
## Demo Video

🐦 [Watch the demo video](https://drive.google.com/file/d/15TDdlnYkVqrAUew5q3ECaeetRxgYMMBE/view?usp=drive_link) 🔗


🚀 **Application:** []()

>>>>>>> 0496fa2789c849142e33c2f428867c54a13291f1

<br />

**Contributing**

Contributions to this project are welcome! If you encounter any issues or have suggestions for improvements, please feel free to submit a pull request.

<br />

<<<<<<< HEAD
=======

**Contact**

📧 Email: palakm130@gmail.com 

🌐 LinkedIn: [linkedin.com/in/palak-mishra-494a2b1b9](https://www.linkedin.com/in/palak-mishra-494a2b1b9)
>>>>>>> 0496fa2789c849142e33c2f428867c54a13291f1

For any further questions or inquiries, feel free to reach out. We are happy to assist you with any queries.

=======
# Bird-Sound-Classification-
A bird sound classification system that identifies bird species from audio recordings using deep learning algorithms.
<<<<<<< HEAD
>>>>>>>
=======
>>>>>>> 

>>>>>>> 0496fa2789c849142e33c2f428867c54a13291f1
