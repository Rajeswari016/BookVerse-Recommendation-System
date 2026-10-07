# 📚 BookVerse – Book Recommendation System

BookVerse is a **Machine Learning-based Book Recommendation System** that provides personalized book recommendations using **Collaborative Filtering and Singular Value Decomposition (SVD)**.

The system analyzes user-book rating data and identifies books that are likely to match a user's interests.

## 🚀 Project Overview

With a large number of books available, finding books that match a reader's interests can be difficult. This project uses historical user ratings to build a recommendation system that suggests relevant books based on user preferences.

### 🎯 Objective

* Analyze user-book rating data.
* Identify user reading preferences.
* Apply collaborative filtering techniques.
* Build an SVD-based recommendation model.
* Generate personalized book recommendations.
* Provide an interactive interface for exploring recommendations.

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Scikit-surprise / SVD**
* **Streamlit**
* **Matplotlib**
* **Jupyter Notebook**

## 📊 Dataset

The project uses three main datasets:

### Books.csv

Contains information about books such as:

* Book ID
* Book title
* Author
* Publisher
* Publication details

### Ratings.csv

Contains user-book rating information:

* User ID
* Book ID
* Rating

### Users.csv

Contains user-related information used during data analysis and recommendation.

## 🔄 Project Workflow

```text
Raw Dataset
     ↓
Data Loading
     ↓
Data Cleaning & Preprocessing
     ↓
Exploratory Data Analysis
     ↓
User-Book Rating Matrix
     ↓
Collaborative Filtering
     ↓
SVD Model
     ↓
Model Training
     ↓
Recommendation Generation
     ↓
Streamlit Application
```

## 🤖 Recommendation Approach

The project uses **Collaborative Filtering**, where recommendations are generated based on user rating patterns.

### SVD – Singular Value Decomposition

SVD is used to learn latent relationships between users and books.

The model learns:

* User preferences
* Book characteristics
* User-book interaction patterns

These learned patterns are then used to predict ratings and generate recommendations.

## 📈 Model

**Algorithm:** Singular Value Decomposition (SVD)

The trained model is used to estimate user preferences and recommend books with higher predicted ratings.

## 🖥️ Application

The project includes a **Streamlit web application** that allows users to interact with the recommendation system.

Users can:

* Search for books
* Explore available books
* Get personalized recommendations
* View book-related information
* Explore model and dataset information

## 📁 Project Structure

```text
BookVerse-AI-Recommendation-System/
│
├── Books.csv
├── Ratings.csv
├── Users.csv
├── Python_File.ipynb
├── app.py
├── requirements.txt
└── README.md
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Rajeswari016/BookVerse-Recommendation-System.git
```

### 2. Navigate to the project directory

```bash
cd BookVerse-Recommendation-System
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

## 💡 Key Features

* 📚 Book search
* ⭐ Rating-based recommendations
* 👤 User preference analysis
* 🔍 Collaborative filtering
* 🧮 SVD-based recommendation model
* 📊 Dataset analysis
* 🌐 Interactive Streamlit interface

## 📌 Project Highlights

* Worked with a large-scale book rating dataset.
* Performed data cleaning and preprocessing.
* Analyzed user-book interactions.
* Implemented collaborative filtering.
* Built an SVD recommendation model.
* Developed an interactive Streamlit application.

## 🔮 Future Improvements

* Add content-based filtering.
* Build a hybrid recommendation system.
* Improve cold-start handling for new users and books.
* Integrate book cover images and external book information.
* Deploy the application online.
* Improve recommendation evaluation using additional metrics.

## 👩‍💻 Author

**Dama Rajeswari**

Computer Science Engineering Graduate
Data Science | Machine Learning | Python | SQL

GitHub: https://github.com/Rajeswari016

LinkedIn: https://linkedin.com/in/rajeswari-dama
