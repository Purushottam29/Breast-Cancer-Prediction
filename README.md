# Breast Cancer Prediction

## Overview
Breast cancer is one of the most common types of cancer worldwide. Early detection is crucial for effective treatment and better survival rates. This project utilizes **Deep Learning** to predict breast cancer based on ultrasound images, leveraging the **DenseNet121** model for feature extraction and classification.

## Features
- **Ultrasound Image Analysis**: Takes ultrasound images as input.
- **Deep Learning Model**: Uses **DenseNet121** for classification.
- **Web-Based Interface**: Built with **Streamlit**.
- **University GPU Support**: Accelerates model training and inference.

## Tech Stack
- **Machine Learning Framework**: TensorFlow/Keras, OpenCV, NumPy
- **Version Control**: Git, GitHub

## Installation
### Clone the Repository
```sh
git clone git@github.com:Purushottam29/Breast-Cancer-Prediction.git
cd Breast-Cancer-Prediction
```
### Set Up Python Environment
```sh
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
### Run the streamlit file
```sh
streamlit run sample.py  # Modify based on your framework
```

## Model Training
If you wish to retrain the model:
```sh
python train.py  # Ensure all dependencies are installed
```

## Usage
1. Upload an ultrasound image via the web interface.
2. After image processing prediction result is returned. 
3. View the results on the web interface.

## Results & Accuracy
The **DenseNet121** model achieves high accuracy in detecting breast cancer. Detailed evaluation metrics are available in the `results/` folder.

## Future Improvements
- Implement **real-time inference**.
- Enhance **explainability** using Grad-CAM.
- Deploy on **cloud platforms**.

## Contributing
Feel free to open issues or submit pull requests. Contributions are welcome!

## License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## Contact
For queries, reach out to [Purushottam](https://github.com/Purushottam29) or email at **purushottamchoudhary2910@gmail.com**.

