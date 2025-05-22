# Student GPA Prediction

This project builds a simple machine learning model to predict the GPA (Grade Point Average) of students based on their academic performance and attendance rate.

## Dataset

- Source: [Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
- Original columns: math score, reading score, writing score, etc.
- Processed features:
  - `assignment_score` – based on reading & writing
  - `attendance_rate` – simulated percentage (75%–100%)
  - `midterm_score` – derived from math score
  - `final_score` – derived from writing score
  - `gpa` – average of assignment, midterm, and final scores

## Technologies

- Python
- Google Colab
- Pandas, NumPy
- scikit-learn
- Matplotlib, Seaborn

## Model

- Model: Linear Regression
- Target: GPA (continuous value between 0 and 10)

## Visualizations

- GPA distribution (Histogram)
- Student performance by category (Pie chart)

## Project Structure

Student-GPA-Prediction/ 
├── processed_students_data.csv # Final dataset used for modeling 
├── Student_GPA_Colab.ipynb # Main notebook (Google Colab) 
├── README.md # Project description

## How to Run

1. Open `Student_GPA_Colab.ipynb` in [Google Colab]
2. Follow the cells from top to bottom:
   - Load dataset
   - Process features
   - Train model
   - Evaluate and visualize results

## Result

- Mean Squared Error: _(based on your training)_
- R² Score: _(based on your training)_

## Future Improvements

- Use real attendance data
- Try other regression models (RandomForest, XGBoost)
- Build web interface (Flask/FastAPI + React)
