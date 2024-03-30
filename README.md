# 🎤 Speech Emotion Recognition using Deep Learning 🔊

Welcome to my GitHub repository showcasing a deep learning experiment on Speech Emotion Recognition (SER)! This project aims to develop a powerful neural network model capable of recognizing emotions from speech audio files. Let's dive into the exciting world of speech processing and emotion detection! 💻🔍

![Alt Text](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTOoK_1HnzLOzSv0oavpHVvoDw2xSYYYK-iYY-a36c6cAtp6oMULFmsOBUPYLlAQmNqluU&usqp=CAU)
![Alt Text](https://camo.githubusercontent.com/ec0e2c310ef7bc717ed94626e8b74f02b5367e16011697d382f45bf996d4158a/68747470733a2f2f692e696d6775722e636f6d2f663154717669542e6a706567)
![Alt Text](https://www.mdpi.com/sensors/sensors-21-05554/article_deploy/html/images/sensors-21-05554-g001.png)
![Alt Text](https://media.springernature.com/m685/springer-static/image/art%3A10.1007%2Fs11042-020-09874-7/MediaObjects/11042_2020_9874_Fig1_HTML.png)

## 📂 Dataset Preparation

To train our model, we gathered several publicly available speech emotion datasets, including:

- 📚 **RAVDESS**: Ryerson Audio-Visual Database of Emotional Speech and Song
- 📖 **TESS**: Toronto Emotional Speech Set
- 📜 **SAVEE**: Surrey Audio-Visual Expressed Emotion
- 📗 **CREMA-D**: Crowd-sourced Emotional Multimodal Actors Dataset

These datasets provide a diverse collection of speech samples labeled with different emotions, ensuring a comprehensive training environment for our model.

## 🔬 Feature Extraction

One of the crucial steps in speech processing is feature extraction. We utilized the powerful `librosa` library to compute **Mel-Frequency Cepstral Coefficients (MFCCs)** from the audio data. MFCCs are widely used and effective features for speech recognition tasks.

```python
import librosa

def extract_features(data):
    mfccs = librosa.feature.mfcc(y=data, sr=22050, n_mfcc=58)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed
```

To further enhance the diversity of our training data, we applied various **data augmentation techniques**, such as adding noise, time stretching, pitch shifting, and speed alteration. These techniques help the model generalize better and improve its robustness.

## 🧠 Neural Network Architecture

At the core of our project lies a powerful **Convolutional Neural Network (CNN)** model built using `tensorflow` and `keras`. The model architecture comprises multiple 1D convolutional layers, pooling layers, dropout layers, and dense layers, carefully designed to capture the intricate patterns present in speech data.

```python
def build_model(in_shape):
    model = Sequential()
    model.add(Conv1D(256, kernel_size=6, activation='relu', input_shape=(in_shape, 1)))
    model.add(AveragePooling1D(pool_size=4, strides=2))
    # ... (additional layers)
    model.add(Dense(units=4, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
```

To ensure comprehensive training and evaluation, we trained separate models for mixed-gender data, female data, and male data.

## 📈 Model Training and Evaluation

During the training process, we leveraged techniques like learning rate reduction on plateau and early stopping to optimize the model's performance. Additionally, we utilized TensorFlow's `MirroredStrategy` to distribute the training across multiple GPUs, if available, accelerating the training process.

```python
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = build_model(x_train.shape[1])
    history = model.fit(x_train, y_train, ...)
```

To assess the model's performance, we calculated accuracy scores for both training and testing data, separately for mixed-gender, female, and male models. Our models achieved impressive results, with the mixed-gender model achieving around **80% testing accuracy**, and the gender-specific models (female and male) achieving around **75% testing accuracy**.

```python
score = model.evaluate(x_test, y_test, verbose=0)
print(f"Testing Accuracy: {score[1]:.2%}")
```

Furthermore, we generated confusion matrices to visualize the model's performance in classifying different emotions, providing valuable insights for further improvements.

## 📊 Results and Insights

Our deep learning experiment on Speech Emotion Recognition has yielded promising results, demonstrating the potential of neural networks in accurately recognizing emotions from speech data. However, there is still room for improvement, and we plan to explore more advanced techniques, such as transfer learning and attention mechanisms, to further enhance the model's performance.

## 🚀 Future Directions

Looking ahead, we envision integrating this Speech Emotion Recognition model into various real-world applications, such as:

- 🎭 Emotion-aware virtual assistants and chatbots
- 🎥 Sentiment analysis in multimedia content
- 📞 Call center analytics for improved customer service
- 🧑‍🏫 Educational tools for emotional intelligence development

We are excited to continue pushing the boundaries of speech processing and emotion recognition, and we welcome contributions, suggestions, and collaborations from the open-source community.

Let's embark on this exciting journey together and unlock the full potential of deep learning in understanding the emotions conveyed through speech! 🎉🔥
