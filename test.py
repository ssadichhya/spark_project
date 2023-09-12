#calculating quartiles for price column. This results in 3 quartiles Q1,Q2 and Q3.
quartiles = listings_df.stat.approxQuantile("price", [0.25, 0.5, 0.75], 0.01)
# quantiles

#creating UDF for categorizing listings into Cheap, Mid-range and Luxury according to price and taking quartiles as limit.

def categorize_price(price):
    if price <= quartiles[0]: #first quartile
        return "cheap"
    elif price <= quartiles[2]: #second quartile
        return "mid-range"
    else:
        return "luxury"

#Registering UDF to spark
categorize_price_udf = f.udf(categorize_price, StringType())

# Create a new column 'price_category' based on the categorization
listings_df_1 = listings_df.withColumn("price_category", categorize_price_udf(f.col("price")))
# listings_df_1.show()

# category_stats = listings_df.groupBy("id").agg({"bedrooms": "sum", "bathrooms": "sum"})

category_stats = listings_df_1.select("id","bedrooms", "bathrooms", "price","price_category")

# duplicate_names_count = listings_df.select("name").groupBy("name").count().filter(f.col("count") > 1)
# duplicate_names_count.show(truncate=False)

# listings_df.filter(f.col("name") == 'Furnished Longwood 1BR Apartment.').show(truncate=False)

# Get the most value for money properties (e.g., highest bedrooms + bathrooms per dollar spent)
# category_stats = category_stats.withColumn("value_for_money",
#                                            (f.col("bedrooms") + f.col("bathrooms")) / f.expr(f"percentile_approx(price, array({quartiles[0]}, {quartiles[2]}))"))
# category_stats.show()

# Create a Window specification to partition data by price category
window_spec = Window.partitionBy("price_category")

# Define a UDF to calculate the percentile
def calculate_percentile(price):
    if price <= quartiles[0]:
        return 25.0
    elif price <= quartiles[1]:
        return 50.0
    else:
        return 75.0

# Register the UDF
calculate_percentile_udf = f.udf(calculate_percentile, DoubleType())

# Calculate the percentile for each row
category_stats = category_stats.withColumn(
    "percentile",
    calculate_percentile_udf(f.col("price"))
)
category_stats.show()

# Calculate the value for money and add it as a new column
# category_stats = category_stats.withColumn(
#     "value_for_money",
#     (f.col("total_bedrooms") + f.col("total_bathrooms")) / (f.col("Q1_price") + f.col("Q3_price"))
# )

# Create a Window specification to partition data by price category
# window_spec = Window.partitionBy("price_category")

# # Define a UDF to calculate the percentile
# def calculate_percentile(price):
#     if price <= quartiles[0]:
#         return 25.0
#     elif price <= quartiles[1]:
#         return 50.0
#     else:
#         return 75.0

# # Register the UDF
# calculate_percentile_udf = f.udf(calculate_percentile, DoubleType())

# Calculate the percentile for each row
# category_stats = category_stats.withColumn(
#     "q1_price",quartiles[0])
# quartiles[0]



# # Find the most value for money category
# window_spec = Window.orderBy(f.col("value_for_money").desc())
# most_value_for_money = category_stats.withColumn("rank", f.row_number().over(window_spec)).filter(f.col("rank") == 1).select("price_category")
# most_value_for_money.show()
 