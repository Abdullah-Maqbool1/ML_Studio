# ML Studio

**A Production-Grade Multi-Model Machine Learning Platform**
*Applied Machine Learning Semester Project - University of Central Punjab*

![ML Studio Architecture](https://github.com/Abdullah-Maqbool1/ML_Studio/blob/main/Screenshots/architecture.png)

---

## Executive Summary

**ML Studio** is a centralized Machine Learning platform that integrates multiple predictive and analytical models into a single web application. The system enables users to upload data, execute machine learning models, and visualize results through an intuitive dashboard.

The platform combines techniques from **Clustering, Computer Vision, Audio & Speech Processing, Natural Language Processing (NLP), and Data Mining**, providing a practical demonstration of concepts learned throughout the Applied Machine Learning course.

---

## Features

### Clustering Pipelines

* **K-Means Clustering**

  * Centroid-based unsupervised clustering
  * Interactive cluster visualization

* **DBSCAN Clustering**

  * Density-based clustering
  * Automatic noise and outlier detection

### Computer Vision

* **Image Classification**

  * CNN-based binary image classification
  * Custom dataset training support

### Audio & Speech Processing

* **Voice Sentiment Analysis**

  * Speech-to-Text conversion
  * Sentiment classification from spoken input

* **Voice Question Answering**

  * Audio input processing
  * Intelligent voice-based response generation

### NLP & Text Processing

* **Text Generation**

  * GPT-2 based text continuation

* **English to Urdu Translation**

  * Transformer-based neural machine translation

* **Named Entity Recognition (NER)**

  * Detection of:

    * Person Names
    * Organizations
    * Locations
    * Miscellaneous Entities

### Data Mining

* **Apriori Association Rule Mining**

  * Frequent itemset generation
  * Support calculation
  * Confidence calculation
  * Lift metric analysis

---

## System Architecture

### Architecture Diagram

![System Architecture](https://github.com/Abdullah-Maqbool1/ML_Studio/blob/main/Screenshots/architecture.png)

### Technology Stack

![Technology Stack](https://github.com/Abdullah-Maqbool1/ML_Studio/blob/main/Screenshots/stack.jpg)

### Core Modules

![Core Modules](https://github.com/Abdullah-Maqbool1/ML_Studio/blob/main/Screenshots/core_modules.jpg)

### Data Flow Pipeline

![Data Flow Pipeline](https://github.com/Abdullah-Maqbool1/ML_Studio/blob/main/Screenshots/dataFlowPipeline.jpg)

---

## Application Screenshots

### Main Interface

![Screenshot 1](https://github.com/Abdullah-Maqbool1/ML_Studio/blob/main/Screenshots/Picture1.png)

![Screenshot 2](https://github.com/Abdullah-Maqbool1/ML_Studio/blob/main/Screenshots/Picture2.png)

![Screenshot 3](https://github.com/Abdullah-Maqbool1/ML_Studio/blob/main/Screenshots/Picture3.png)

![Screenshot 4](https://github.com/Abdullah-Maqbool1/ML_Studio/blob/main/Screenshots/Picture4.png)

![Screenshot 5](https://github.com/Abdullah-Maqbool1/ML_Studio/blob/main/Screenshots/Picture5.png)

For additional screenshots, visit the **Screenshots** folder in the repository.

---

## Technology Stack

### Frontend

* HTML5
* CSS3
* JavaScript
* Jinja2 Templates

### Backend

* Python
* Flask

### Machine Learning Libraries

* Scikit-Learn

  * K-Means
  * DBSCAN
  * Apriori

* TensorFlow / PyTorch

  * Convolutional Neural Networks (CNN)

* Hugging Face Transformers

  * GPT-2
  * Translation Models
  * NER Models

### Deployment

* Flask Local Server

---

## Project Structure

```bash
ML_Studio/
│
├── app.py
├── models/
├── cnn_datasets/
├── transformer_text_generation/
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── templates/
├── Screenshots/
└── README.md
```

---

## Installation & Setup

### Clone Repository

```bash
git clone https://github.com/Abdullah-Maqbool1/ML_Studio.git
cd ML_Studio
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

### Open in Browser

```text
http://127.0.0.1:5000
```

---


## Future Enhancements

* User Authentication & Authorization
* Real-Time Analytics Dashboard
* Model Retraining Through Web Interface
* Docker Containerization
* Cloud Deployment Support
* Automated Performance Monitoring
* PDF Report Generation
* Experiment Tracking Dashboard

---

## Learning Outcomes

This project demonstrates practical implementation of:

* Unsupervised Learning
* Deep Learning
* Natural Language Processing
* Speech Processing
* Data Mining
* Machine Learning Deployment
* Full-Stack ML Application Development

---

## License

This project was developed for educational purposes as part of the **Applied Machine Learning** course at the **University of Central Punjab (UCP)**.

---

### Made with ❤️ for Applied Machine Learning
