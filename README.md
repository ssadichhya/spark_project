# Airbnb Property Analysis

In this project, we analyze Airbnb property data to answer two questions:

## Question 1: Property Price Categories and Value for Money

### Overview
We categorize Airbnb properties into three categories: cheap, mid-range, and luxury, based on their prices using quartiles. We then analyze the total number of bedrooms and bathrooms in each category and find the most value-for-money properties. Additionally, we perform sentiment analysis on reviews to provide ratings for each property.

### Steps
1. Categorize properties into cheap, mid-range, and luxury based on prices.
2. Analyze the total number of bedrooms and bathrooms in each category.
3. Calculate the value-for-money metric for each property.
4. Perform sentiment analysis on property reviews to provide ratings.
5. Identify the most value-for-money properties and their sentiment analysis ratings.

## Question 2: Booking Patterns and Host Analysis

### Overview
We determine the month with the highest number of bookings by using quartiles to define peak and off-peak times. We then list properties in both peak and off-peak times. Additionally, we calculate the total revenue earned by each host during peak months and analyze their average response rates to find correlations.

### Steps
1. Identify the month with the highest number of bookings using quartiles.
2. Categorize properties into peak and off-peak times based on booking patterns.
3. List properties available during peak and off-peak times.
4. Calculate the total revenue generated by each host during peak months.
5. Analyze the average response rate of hosts during peak months.
6. Find correlations between total revenue and average response rate.

## Data Sources
- Airbnb property data, including price, bedrooms, bathrooms, reviews, and host information.
- Calendar data with booking information.

## Technologies Used
- Python
- PySpark
- Sentiment analysis libraries (e.g., TextBlob)

## How to Run the Code
1. Ensure you have the necessary libraries and dependencies installed (e.g., PySpark, TextBlob).
2. Run the provided code for each question (Q1 and Q2) to perform the analysis.
3. Review the results and insights provided in the code outputs.


