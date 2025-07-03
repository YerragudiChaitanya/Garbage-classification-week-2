🗑️ Garbage Classification using Transfer Learning (EfficientNetV2B2)






📌 Project Overview




This project builds a deep learning model for automated garbage classification using EfficientNetV2B2 and Transfer Learning. The goal is to facilitate waste segregation at source, supporting environmental sustainability by improving recycling efficiency.



🎯 Objective



To develop an accurate image classification model that categorizes waste images into their respective garbage classes using a pre-trained CNN model.




🧰 Tech Stack


Python 3.10+


TensorFlow / Keras


EfficientNetV2B2 (pre-trained)


Matplotlib, Seaborn, Scikit-learn


Google Colab / Jupyter Notebook



📁 Dataset



Dataset used: archive (9).zip




Structure after extraction:

garbage_data/



└── Garbage classification/



    └── Garbage classification/


    
        ├── cardboard/


        
        ├── glass/


        
        ├── metal/


        
        ├── paper/


        
        ├── plastic/


        
        └── trash/

Each folder contains labeled images for the respective waste category.

🧪 Model Architecture

Base Model: EfficientNetV2B2 (with pre-trained ImageNet weights)

Custom Head:

GlobalAveragePooling2D

Dropout (0.3)

Dense output layer (softmax activation)

Loss Function: Sparse Categorical Crossentropy

Optimizer: Adam (learning rate: 0.0001)

🧹 Data Pipeline

Image size: 260x260

Batch size: 32

Dataset split:

80% Training

10% Validation

10% Test (from validation split)

Datasets are cached and prefetched for performance.

📊 Model Training & Evaluation

Training: Performed for 10 epochs

Metrics Monitored:

Training & Validation Accuracy

Training & Validation Loss

📈 Evaluation Metrics:

Confusion Matrix

Classification Report (Precision, Recall, F1-score)

Test Accuracy: 0.8398

Test Accuracy: 83.98%

Test Loss: 0.6021

🔍 Visualization:

Accuracy & Loss Curves

Sample Predictions from the test set

💾 Model Saving

The trained model is saved using the Keras v3 format:

model.save("efficientnetv2b2_garbage_classifier.keras")

✅ How to Run

Upload and extract dataset (archive (9).zip)

Run all cells in order in Garbage_classification_week_2.ipynb

Evaluate final model performance

Save or deploy the model as required

🚀 Future Enhancements

Add image prediction upload interface

Deploy model via Streamlit or Flask

Convert to TensorFlow Lite for mobile apps

👨‍💻 Author

Project by: [Your Name]

College: [Your College Name]

GitHub: https://github.com/YourGitHubUsername

📜 License

This project is licensed under the MIT License.
