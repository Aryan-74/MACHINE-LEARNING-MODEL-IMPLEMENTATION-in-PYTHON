# MACHINE-LEARNING-MODEL-IMPLEMENTATION-in-PYTHON

this is the fourth project of my first summer internship in python programming

COMPANY: CODTECH IT SOLUTIONS

NAME: Aryan Jain

INTERN ID: CT04DG1236

DOMAIN: PYTHON PROGRAMMING

DURATION: 4 WEEKS

DESCRIPTION:

**Overview:**

Task 4 of the CODTECH Python Internship involves designing a **predictive machine learning model** using the **scikit-learn** library to classify or predict outcomes from a dataset. The goal is to introduce interns to the practical application of machine learning — from loading data and cleaning it, to training a model, testing it, and interpreting the results. For this task, a **Spam Email Detection Model** was developed using a real-world dataset and a classic text classification algorithm: **Multinomial Naive Bayes**.



**Objective:**

The main objective of this task is to simulate a real-world application of machine learning where the goal is to identify whether a given message is **spam** or **ham** (not spam). This is one of the most common and useful NLP (Natural Language Processing) classification tasks. Interns were expected to:

* Load and explore a dataset
* Preprocess the data
* Convert text into numerical form
* Train a classification model
* Evaluate its performance using common metrics
* Test custom messages for prediction

The task emphasizes understanding both the theoretical and practical aspects of machine learning workflows.



**Approach:**

For this task, the open-source **SMS Spam Collection Dataset** was used. This dataset contains over 5,000 messages labeled as either `ham` or `spam`. The following steps were followed in the development of the model:

1. **Data Loading and Exploration**:
   The dataset was loaded using **pandas**. The first step was to examine the structure of the data, understand the number of spam vs ham messages, and check for any missing values.

2. **Data Preprocessing**:
   The labels (`ham`, `spam`) were converted into numerical format (0 and 1 respectively). This made it possible for the model to learn from the data.

3. **Text Vectorization**:
   Since machine learning models require numerical inputs, the message texts were converted into numeric vectors using the **CountVectorizer**. This transforms the words into token counts for each message — creating a “bag of words” representation.

4. **Splitting Data**:
   The dataset was split into training and testing sets (typically 75% training and 25% testing) using `train_test_split` from scikit-learn. This ensures that the model is trained on one portion and evaluated on unseen data.

5. **Model Training**:
   A **Multinomial Naive Bayes** classifier was chosen, as it is highly effective for text classification tasks, especially spam detection. The model was trained on the vectorized training messages.

6. **Model Evaluation**:
   The model was then tested on the test set, and its accuracy, precision, recall, and F1-score were calculated using `accuracy_score` and `classification_report`. Additionally, a **confusion matrix** was generated and visualized using **Seaborn** to better understand prediction performance.

7. **Custom Prediction**:
   A function was implemented to allow users to input any message and predict whether it would be classified as spam or ham. This made the model interactive and practical for end users.



**Learning Outcomes:**

This task provided hands-on experience in the complete **machine learning pipeline**, from data loading to model deployment. Interns gained insight into:

* Text classification and spam filtering techniques
* Natural Language Processing with `CountVectorizer`
* The importance of data preprocessing and vectorization
* Using `MultinomialNB` for NLP tasks
* Performance evaluation using metrics and confusion matrix
* Creating reusable prediction functions

It also strengthened the intern’s understanding of **data science workflows**, and how ML models are tested, tuned, and interpreted in real-life applications.



**Conclusion:**

Task 4 offered a well-rounded introduction to real-world machine learning using Python. By implementing a spam detection system, interns not only applied ML concepts but also saw how models can be used to make meaningful predictions on real-life data. The use of scikit-learn streamlined the process, while the combination of `pandas`, `matplotlib`, and `seaborn` added value in terms of data analysis and visualization. By the end of the task, the interns were able to confidently build, evaluate, and use a text classification model — a skill highly valuable in both academic and professional settings.

SMS SPAM DATASET : <img width="451" height="243" alt="image" src="https://github.com/user-attachments/assets/1b3abeb5-b950-494d-8e30-1cebc89e5d82" />

OUTPUT : <img width="531" height="263" alt="image" src="https://github.com/user-attachments/assets/2ae6f7c9-cb4b-4d03-b165-f07b7e7a0510" />

<img width="675" height="556" alt="image" src="https://github.com/user-attachments/assets/5c4a9127-dcf8-4c8b-ae0a-968f8280a9e2" />



