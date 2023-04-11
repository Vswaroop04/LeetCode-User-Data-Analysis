# LeetCode-User-Data-Analysis
We have Created a Datascience Life Cycle Which Involves

1. Data Collection
2. Data Cleaning
3. EDA
4. Model Building
5. Model Deployment

Our Application Deployed at [Streamlit](https://vivek-borole-leetcode-data-analysis-new-hn2yrb.streamlit.app/)

# Project: Leetcode User Data Analytics

We have done Data analysis on Leetcode Users data such as  Total number of attempted questions, number of Easy/Medium/Hard questions attempted, Name, Country, PostViewCount, SolutionCount, Reputation, ActiveYears, Streak, TotalActiveDays, AttendedContestsCount, Rating, GlobalRanking, TotalParticipants, TopPercentage, and Badges earned.

## Dataset

### Input
The input dataset for this project is taken from [kaggle](https://www.kaggle.com/datasets/nidhaypancholi/leetcode-indian-user-ratings) file which contains leetcode indian usernames of around 170000

-- We have taken this input file and created an api to fetch the required data

#### Creation of API
Firstly we found graphql api of Leetcode in which they are using query response method to fetch data
   [img](https://i.ibb.co/2YKzXNv/api.png)
The Api we had created involves running 5,6 Queries of this graphql api 


### How to Run

To Run Go to the src/codeforgeneratingdataset

1. Run npm install
2. Run node app.js
3. Send API Response to `http://localhost:3000/api` and Check Logs for the progress

## Output
The output of this Api ncludes various features for each user, such as Sno, Username, Total number of attempted questions, number of Easy/Medium/Hard questions attempted, Name, Country, PostViewCount, SolutionCount, Reputation, ActiveYears, Streak, TotalActiveDays, AttendedContestsCount, Rating, GlobalRanking, TotalParticipants, TopPercentage, and Badges earned.

## Data Analysis

We have Used Plotly to plot the graphs and used Streamlit For analysis Our analysis part is located at `Analysis\Untitled.ipynb` 

### How to Run

-- Install the Dependencies using `pip install <dependency_name>`

1. Streamlit
2. Streamlit-lottie
3. xgboost
4. Plotly
5. Pandas, Numpy, Matplotlib, Seaborn, PIL, io and time

-- Now After Installing, Run  streamlit run '.\Leetcode Data Analysis.py'

Then, you can see the data analysis in an application form at `http://localhost:8501`

The data analysis includes:
- Exploratory data analysis (EDA) to gain insights into the distribution and relationships between the variables
- Correlation analysis to identify the strength and direction of the relationships between the variables
- Regression analysis to model the relationships between the variables and make predictions based on the models
- Data visualization to present the findings of the analysis in an easy-to-understand format.
- 3d Plots using Clustering


Team Members Involved :-

1. K V Vishnu Swaroop
2. Rohith Khandal
3. Ishant Bisen
4. Vivek Borole
5. Gurupal Singh

-------------------------------------------------------------------------   Made by Vishnu ----------------------------------------------------------------------------
