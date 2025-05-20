# Netflix Titles – Exploratory Data Analysis (EDA)

This project explores a dataset of Netflix Movies and TV Shows using Python and Jupyter Notebook. It includes data cleaning, visualization, and key insights on the distribution of content across type, country, year, rating, and genre.

## Dataset

- Source: [Kaggle - Netflix Movies and TV Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows)
- Records: ~8,800 titles
- Features: title, director, cast, country, date added, release year, rating, genre, etc.

## Objectives

- Analyze the distribution of Movies vs TV Shows
- Identify the top content-producing countries
- Explore trends over time (year added)
- Understand the most common ratings and genres

## Key Insights

- Movies make up the majority of content on Netflix.
- The United States is the top producer, followed by India and the UK.
- Content uploads peaked between 2016 and 2020.
- The most frequent rating is TV-MA.
- Documentaries, Dramas, and Comedies are the most common genres.

## Tools and Libraries

- Python
- Pandas
- Matplotlib
- Seaborn
- Google Colab / Jupyter Notebook

## Folder Structure
netflix-eda/
├── netflix_eda.ipynb       # Main Jupyter notebook 
├── README.md               # Project description
├── datasets/
│   └── netflix_titles.csv  # Raw dataset


## How to Run

1. Open the notebook in Google Colab or Jupyter Notebook.
2. Upload `netflix_titles.csv` or load it from your local/dataset path.
3. Run all cells in order to view the analysis and charts.
4. (Optional) Save charts as `.png` to use in the README or documentation.

## Future Improvements

- Compare with other streaming platforms (e.g., Prime Video, Disney+)
- Use NLP techniques to analyze movie/TV show descriptions
- Apply time-series analysis to forecast content trends

## License

This project is for educational and personal portfolio use. Dataset provided by Kaggle (publicly available).

