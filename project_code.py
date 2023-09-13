# %%
from pyspark.sql import SparkSession
from pyspark.sql.functions import when
from pyspark.sql import functions as f 
from pyspark.sql.window import Window
from pyspark.sql.types import StringType,IntegerType
from dotenv import load_dotenv
from textblob import TextBlob
import os 
import yaml

# Define the path to your YAML file
yaml_file_path = 'config.yaml'

# Read the YAML file and parse it into a Python dictionary
with open(yaml_file_path, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)


# %%
# Initialize a Spark session
spark = SparkSession \
    .builder \
    .appName("Day_6") \
    .config("spark.jars", "/usr/lib/jvm/java-11-openjdk-amd64/lib/postgresql-42.6.0.jar") \
    .getOrCreate()

# %% [markdown]
# ## Data Cleaning

# %% [markdown]
# **Extract the data**

# %%
# Load the CSV data into DataFrames
listing_df_raw = spark.read.csv('raw_data/listings.tsv', header=True, inferSchema=True, sep="\t")
reviews_df_raw = spark.read.csv('raw_data/reviews.tsv', header=True, inferSchema=True, sep="\t")
calendar_df_raw = spark.read.csv('raw_data/calendar.tsv', header=True, inferSchema=True,sep="\t")

# %% [markdown]
# **Transform the data**

# %%
# Replace commas in all columns using a loop
for column in listing_df_raw.columns:
    listing_df_raw = listing_df_raw.withColumn(column, f.regexp_replace(f.col(column), ',', ''))

# Transform the data

# LISTINGS DATA

# Drop the 'summary' column
listing_df_raw = listing_df_raw.drop('summary')
listing_df_raw = listing_df_raw.drop('description')
listing_df_raw = listing_df_raw.drop('host_about')


# Convert 'host_is_superhost' to boolean
listing_df_raw = listing_df_raw.withColumn('host_is_superhost', when(f.col('host_is_superhost') == 't', True).otherwise(False))

# Drop 'country' and 'market' columns
listing_df_raw = listing_df_raw.drop('country', 'market')

# Drop rows with null values in the 'space' column
listing_df_raw = listing_df_raw.na.drop(subset=['space'])\


# Remove "$" and convert to integer
listing_df_raw = listing_df_raw.withColumn("price", f.regexp_replace(f.col("price"), "[^0-9]", "").cast(IntegerType()))



# CALENDAR DATA

# Convert 'available' to boolean
calendar_df_raw = calendar_df_raw.withColumn('available', when(f.col('available') == 't', True).otherwise(False))

#fill price colum with "booked"
calendar_df_raw = calendar_df_raw.withColumn("price", when(f.col("available") == False, "booked").otherwise(f.col("price")))

calendar_df_raw = calendar_df_raw.withColumn("price", f.regexp_replace(f.col("price"), "[^0-9]", "").cast(IntegerType()))


# %% [markdown]
# **Load the data**

# %%
# Save data locally

listing_df_raw.coalesce(1).write.parquet('cleaned_data/clean_listing', mode="overwrite",compression="snappy")
calendar_df_raw.coalesce(1).write.parquet('cleaned_data/clean_calendar',  mode="overwrite",compression="snappy")
reviews_df_raw.coalesce(1).write.parquet('cleaned_data/clean_reviews',  mode="overwrite" ,compression="snappy")

# %%
#load data to postgres

# Define the JDBC connection properties
jdbc_url = "jdbc:postgresql://localhost:5432/spark_project"
properties = {
    "user":config['postgres']["user"],
    "password":config['postgres']["password"],
    "driver": "org.postgresql.Driver"
}
listing_df_raw.show()
# Write the DataFrame to the existing PostgreSQL table
listing_df_raw.write.jdbc(url=jdbc_url, table='listing_table', mode="overwrite", properties=properties)
calendar_df_raw.write.jdbc(url=jdbc_url, table='calendar_table', mode="overwrite", properties=properties)
reviews_df_raw.write.jdbc(url=jdbc_url, table='reviews_table', mode="overwrite", properties=properties)

# %% [markdown]
# ## Analysing

# %% [markdown]
# **Extract Data From Postgres**

# %%
# Read tables from postgres  to df
listings_df = spark.read.format("jdbc").option("url", "jdbc:postgresql://localhost:5432/spark_project") \
    .option("driver", "org.postgresql.Driver").option("dbtable", "listing_table") \
    .option("user",config['postgres']["user"]).option("password",config['postgres']["password"]).load()

calendar_df = spark.read.format("jdbc").option("url", "jdbc:postgresql://localhost:5432/spark_project") \
    .option("driver", "org.postgresql.Driver").option("dbtable", "calendar_table") \
    .option("user", config['postgres']["user"]).option("password", config['postgres']["password"]).load()

reviews_df = spark.read.format("jdbc").option("url", "jdbc:postgresql://localhost:5432/spark_project") \
    .option("driver", "org.postgresql.Driver").option("dbtable", "reviews_table") \
    .option("user", config['postgres']["user"]).option("password",config['postgres']["password"]).load()

# %% [markdown]
# **Q1) Property Price Categories and Value for Money:**
# 
# Divide properties into cheap, mid, and luxury categories based on prices, analyze total bedrooms and bathrooms, and find value-for-money properties along with its sentiment analysis ratings.
# 

# %%
#UDF for sentimantal analysis

def analyze_sentiment(comment):
    if comment is None or not isinstance(comment, str):
        # Handle cases where 'comment' is None or not a string
        return "unknown"
    
    analysis = TextBlob(comment)
    # Classify sentiment as positive, neutral, or negative based on polarity
    if analysis.sentiment.polarity > 0:
        return "positive"
    elif analysis.sentiment.polarity == 0:
        return "neutral"
    else:
        return "negative"

# Register the UDF
sentiment_analysis_udf = f.udf(analyze_sentiment, StringType())

reviews_sentiment = reviews_df.withColumn("sentiment", sentiment_analysis_udf(f.col("comments")))

# reviews_sentiment.show(50, truncate=False)
sentiment_df=reviews_sentiment.select("listing_id","id","sentiment")


#calculating quartiles for price column. This results in 3 quartiles Q1,Q2 and Q3.
quartiles = listings_df.stat.approxQuantile("price", [0.25, 0.5, 0.75], 0.01)

listings_df_1 = listings_df.withColumn(
    "price_category",
    when(f.col("price") <= quartiles[0], "cheap")
    .when((f.col("price") > quartiles[0]) & (f.col("price") <= quartiles[1]), "mid-range")
    .otherwise("luxury")
)

category_stats = listings_df_1.select("id","name","bedrooms", "bathrooms", "price","price_category","host_name","number_of_reviews")

# Calculate value for money as bedrooms + bathrooms per dollar spent
category_stats = category_stats.withColumn(
    "value_for_money",
    (f.col("bedrooms") + f.col("bathrooms")) / f.col("price")
)

window_spec=Window.partitionBy(f.col("price_category")).orderBy(f.col("value_for_money").desc())
most_value_for_money = category_stats.withColumn("rank", f.row_number().over(window_spec)).filter(f.col("rank") <=5 ).select("*")
# most_value_for_money.show()

most_value_for_money = most_value_for_money.withColumnRenamed("id", "listing_id")

listings_with_sentiment = most_value_for_money.join(sentiment_df, most_value_for_money["listing_id"] == sentiment_df["listing_id"], how="left")
listings_with_sentiment=listings_with_sentiment.drop(sentiment_df["listing_id"])
# listings_with_sentiment.printSchema()
listings_with_sentiment_1 = listings_with_sentiment.groupBy("listing_id").agg(f.collect_list("sentiment").alias("review_sentiments"),
                                                                              f.first("name").alias("name"),
                                                                              f.first("bedrooms").alias("bedrooms"), 
                                                                              f.first("bathrooms").alias("bathrooms"),
                                                                              f.first("price").alias("prince"),
                                                                              f.first("price_category").alias("price_category"),
                                                                              f.first("rank").alias("rank"))

print("\nRatings According to Sentimental Analysis for Most Value for Money Properties:")
listings_with_sentiment_1=listings_with_sentiment_1.orderBy(f.col("price_category"))
listings_with_sentiment_1=listings_with_sentiment_1.drop(listings_with_sentiment_1["listing_id"])

listings_with_sentiment_1.show()

# %% [markdown]
# **Load question_1 output**

# %%
# Define the JDBC connection properties
jdbc_url = "jdbc:postgresql://localhost:5432/spark_project"
properties = {
    "user": config['postgres']["user"],
    "password":config['postgres']["password"],
    "driver": "org.postgresql.Driver"
}
listings_with_sentiment_1.write.jdbc(url=jdbc_url, table='question_1_table', mode="overwrite", properties=properties)

# %% [markdown]
# **Q2) Total revenue analysis during peak months** 
# 
# Find out which month has the most booking(Use quartile to get threshold values to determine the off peak and the peak time) Use these values to list out the properties in peak time and off peak time. Then calculate the total revenue of each host during peak months and calculate their average response rate as well to find correlation between them.

# %%
# Calculate booking counts by month
booking_counts_df = calendar_df.groupBy(f.month("date").alias("month")).agg(f.count("*").alias("booking_count")).orderBy(f.col("booking_count"))
result_df = booking_counts_df.join(calendar_df, (f.month(calendar_df["date"]) == booking_counts_df["month"]), "left")

# Calculate quartiles for booking counts
quartiles = result_df.approxQuantile("booking_count", [0.25, 0.75], 0.001)
q1 = quartiles[0]
q3 = quartiles[1]


#Create a new column in result_df to categorize months
result_df = result_df.withColumn("month_category", (f.col("booking_count") >= q1) & (f.col("booking_count") <= q3))


# Join with listings_df based on the id
result_df_1 = listings_df.join(result_df, (listings_df["id"]) == result_df["listing_id"], "left")
result_df_1 = result_df_1.withColumn(
    "month_category",
    f.when(f.col("month_category") == True, "peak").otherwise("off_peak")
)


result_df_1 = result_df_1.withColumn("revenue", f.when(result_df["available"] == "false", listings_df["price"]).otherwise(0))
# result_df_1.filter(f.col("available") == False).show()
result_df_1 = result_df_1.withColumn("host_response_rate", f.regexp_replace(f.col("host_response_rate"), "%", "").cast("int"))
# result_df_1.show()


total_revenue_by_host = result_df_1.filter(f.col("month_category") == 'peak').groupBy("host_name").agg(f.sum("revenue").alias("total_revenue"),f.coalesce(f.avg("host_response_rate"),f.lit(0)).alias("avg_response_rate"))
# total_revenue_by_host.show()

correlation = total_revenue_by_host.corr("total_revenue", "avg_response_rate")


total_revenue_by_host.show()
print("the correlation between total revenue and average response rate:", correlation)

# %% [markdown]
# **load question_2 output**

# %%
# Define the JDBC connection properties
jdbc_url = "jdbc:postgresql://localhost:5432/spark_project"
properties = {
    "user": config['postgres']["user"],
    "password": config['postgres']["password"],
    "driver": "org.postgresql.Driver"
}

total_revenue_by_host.write.jdbc(url=jdbc_url, table='question_2_table', mode="overwrite", properties=properties)


# %%
spark.stop()


