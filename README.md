# emotion-classifier-ml-project

Using the dataset:
https://huggingface.co/datasets/dair-ai/emotion

## Dataset and Licensing

This project uses the **dair-ai/emotion** dataset for educational and research purposes only.

Dataset source:  
https://huggingface.co/datasets/dair-ai/emotion

### Citation
If you use this dataset, please cite:

Saravia, E., Liu, H.-C. T., Huang, Y.-H., Wu, J., & Chen, Y.-S. (2018).  
**CARER: Contextualized Affect Representations for Emotion Recognition**.  
Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing,  
pp. 3687–3697.  
https://www.aclweb.org/anthology/D18-1404

Classes included:
- sadness  
- joy
- love
- anger
- fear
- surprise

## Training the Model ('train_model.py')

This script trains an emotion classification model using TF-IDF vectorisation and Logistic Regression.

### What the script does
- Loads the cleaned dataset specified in 'config.py'
- Splits data into training and test sets (80/20, stratified)
- Converts text into numerical features using 'TfidfVectorizer'
- Trains a Logistic Regression classifier with class balancing
- Evaluates the model (accuracy, precision, recall, F1)
- Logs a classification report and confusion matrix
- Saves the trained model and vectorizer to 'model.pkl'

### How to run
```bash
python src/train_model.py

