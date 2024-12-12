import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Load the Excel file
input_file = "datasets/Carbon_(CO2)_Emissions_by_Country.xlsx"
output_file = "formatted_file.csv"

# Read the Excel file
df = pd.read_excel(input_file)
print(df)

# Save it as a CSV
df.to_csv(output_file, index=False)
print(f"File successfully converted to {output_file}")

# ---  Some operations on Dataset  ---   

# Get a count of rows
print(len(df))   # 5677

# Get the size (rows, columns)
print(f"The number of rows is: {df.shape[0]}\nThe number of columns is: {df.shape[1]}")

# Get data types of the columns
data_types = df.dtypes
print(data_types)


""" 
What is the range of years or time period covered in the data?
"""
from datetime import datetime as dt

df['Date'] = df['Date'].dt.strftime('%Y')   # Formatting the 'Date' column

start_year = df["Date"].min()
end_year = df["Date"].max()
print(f"The dataset covers the time period from {start_year} to {end_year}")


# Group the dataset by years to see how many records exist for each year
year_counts = df["Date"].value_counts().sort_values()
print(year_counts)

# Reorder the dataset by "Year" from min to max
df_sorted = df.sort_values(by="Date")
df_sorted.reset_index(drop=True, inplace=True)
print(df_sorted)

# Convert to DataFrame
data_frame = pd.DataFrame(df)

# Aggregate emissions by region to find which contributes most to global emissions
region_emissions = df.groupby("Region")["Kilotons of Co2"].sum().reset_index()

# Save aggregated data to a text file with column headers
with open("region_emissions.txt", "w", encoding='utf-8') as outfile:
    outfile.write(region_emissions.to_string(index=False, header=True))
    
# Save it as a CSV
output_csv_data = "output.csv"


""" 
Code focues on the "Asia" region, 
sorts countries within it by CO2 emissions in descending order, and extracts the top emitters.

This is useful for identifying the most significant contributors to CO2 emissions within a particular region.
"""
# Analyze top 5 countries within the largest-emitting region
top_countries = df[df["Region"] == "Asia"].sort_values(by="Kilotons of Co2", ascending=False)
print(top_countries.head())


with open("top_countries.csv", "w", encoding='utf-8') as top_country:
    top_country.write(top_countries.head().to_string(index=False, header=True))

# Save it as a CSV
top_country_container = "./top_countries.csv"


"""
1) What are the trends in CO2 emissions over the years?
"""

dataset_3 = pd.read_csv("formatted_file.csv")
dataset_3['Date'] = pd.to_datetime(dataset_3['Date'])
dataset_3["Year"] = dataset_3["Date"].dt.year
yearly_emissions = dataset_3.groupby('Year')["Kilotons of Co2"].sum().reset_index()
print(yearly_emissions)


# Visualize Trends
plt.figure(figsize=(10, 6))
plt.plot(yearly_emissions["Year"], yearly_emissions["Kilotons of Co2"], marker="o", color="b", label="CO2 Emissions")
plt.title("Trends in CO2 Emissions Over the Years")
plt.xlabel("Year")
plt.ylabel("Kilotons of CO2")


# Find the year with the maximum CO2 emissions
max_year = yearly_emissions.loc[yearly_emissions["Kilotons of Co2"].idxmax()]

# Annotate the maximum point
plt.annotate(f"Max: {max_year['Kilotons of Co2']} in {max_year['Year']}",
             xy=(max_year["Year"], max_year["Kilotons of Co2"]),
             xytext=(max_year["Year"] + 1, max_year["Kilotons of Co2"] + 500),
             arrowprops=dict(facecolor='red', arrowstyle='->'))


# Add grid and legend
plt.grid(True)
plt.legend()
plt.show()


""" 
2) Which country in Asia has the highest CO2 emissions over time?

"""
# Filter the dataset for the Asia region
asia_data = df[df["Region"] == "Asia"]

# Group by country and calculate the total CO2 emissions
asia_emissions = asia_data.groupby("Country")["Kilotons of Co2"].sum().reset_index()

# Sort the countries by total CO2 emissions in descending order
asia_emissions = asia_emissions.sort_values(by="Kilotons of Co2", ascending=False)



"""
Create a new column for the formatted strings while keeping the original column numeric:

'Kilotons of Co2' remains numeric for further calculations.

'Formatted Co2' is used only for displaying.

"""
# Create a formatted display column
asia_emissions["Formatted Co2"] = asia_emissions["Kilotons of Co2"].map('{:,.2f}'.format)

# Display the (DataFrame) - the top emitters
print(asia_emissions[["Country", "Kilotons of Co2", "Formatted Co2"]].head(10))

# Check the type and contents of the Kilotons of Co2 column:
print(asia_emissions["Kilotons of Co2"].dtype)

with open("top_emissions_in_asia.txt", "w", encoding='utf-8') as emission_asia_file:
    emission_asia_file.write(asia_emissions[["Country", "Kilotons of Co2", "Formatted Co2"]].head(10).to_string(index=False, header=True))
    emission_asia_file.close()


# Visualize the Top (10) Emittiers  - Using Matplotlib
top_emitters = asia_emissions.head(10)

# Create a bar chart
plt.figure(figsize=(12, 6))
plt.bar(top_emitters["Country"], top_emitters["Kilotons of Co2"], color="skyblue")
plt.title("Top 10 CO2 Emitters in Asia")
plt.xlabel("Country")
plt.ylabel("Total CO2 Emissions (Kilotons)")
plt.xticks(rotation=45)
plt.show()


highest_emitter = asia_emissions.iloc[0]
print(f"The country with the higest CO2 emissions is {highest_emitter['Country']} with {highest_emitter['Kilotons of Co2']} kilotons.")

# asia_data = df[df["Region"] == "Asia"]
africa_data = df[df["Region"] == "Africa"]
americas_data = df[df["Region"] == "Americas"]
oceania_data = df[df["Region"] == "Oceania"]
europe_data = df[df["Region"] == "Europe"]


while True:
    # if africa_data:
    africa_emissions = africa_data.groupby("Country")["Kilotons of Co2"].sum().reset_index()

    africa_emissions = africa_emissions.sort_values(by="Kilotons of Co2", ascending=False)

    africa_emissions["Formatted Co2_Africa"] = africa_emissions["Kilotons of Co2"].map('{:,.2f}'.format)

    print(africa_emissions[["Country", "Kilotons of Co2", "Formatted Co2_Africa"]].head(10))

    with open("top_emissions_in_africa.txt", "w", encoding='utf-8') as emission_africa_file:
        emission_africa_file.write(africa_emissions[["Country", "Kilotons of Co2", "Formatted Co2_Africa"]].head(10).to_string(index=False, header=True))
        emission_africa_file.close()

    top_emitters_africa = africa_emissions.head(10)

    plt.figure(figsize=(12, 6))
    plt.bar(top_emitters_africa["Country"], top_emitters_africa["Kilotons of Co2"], color="magenta")
    plt.title("Top 10 CO2 Emitters in Africa")
    plt.xlabel("Country")
    plt.ylabel("Total CO2 Emissions (Kilotons)")
    plt.xticks(rotation=45)
    plt.show()

    highest_emitter_africa = africa_emissions.iloc[0]
    print(f"The country with the higest CO2 emissions is {highest_emitter_africa['Country']} with {highest_emitter_africa['Kilotons of Co2']} kilotons.")


# elif americas_data:
    americas_emissions = americas_data.groupby("Country")["Kilotons of Co2"].sum().reset_index()

    americas_emissions = americas_emissions.sort_values(by="Kilotons of Co2", ascending=False)

    americas_emissions["Formatted Co2_Americas"] = americas_emissions["Kilotons of Co2"].map('{:,.2f}'.format)

    print(americas_emissions[["Country", "Kilotons of Co2", "Formatted Co2_Americas"]].head(10))

    with open("top_emissions_in_americas.txt", "w", encoding='utf-8') as emission_americas_file:
        emission_americas_file.write(americas_emissions[["Country", "Kilotons of Co2", "Formatted Co2_Americas"]].head(10).to_string(index=False, header=True))
        emission_americas_file.close()

    top_emitters_americas = americas_emissions.head(10)

    plt.figure(figsize=(12, 6))
    plt.bar(top_emitters_americas["Country"], top_emitters_americas["Kilotons of Co2"], color="red")
    plt.title("Top 10 CO2 Emitters in Americas")
    plt.xlabel("Country")
    plt.ylabel("Total CO2 Emissions (Kilotons)")
    plt.xticks(rotation=45)
    plt.show()

    highest_emitter_americas = americas_emissions.iloc[0]
    print(f"The country with the higest CO2 emissions is {highest_emitter_americas['Country']} with {highest_emitter_americas['Kilotons of Co2']} kilotons.")


# elif oceania_data:
    oceania_emissions = oceania_data.groupby("Country")["Kilotons of Co2"].sum().reset_index()

    oceania_emissions = oceania_emissions.sort_values(by="Kilotons of Co2", ascending=False)

    oceania_emissions["Formatted Co2_Oceania"] = oceania_emissions["Kilotons of Co2"].map('{:,.2f}'.format)

    print(oceania_emissions[["Country", "Kilotons of Co2", "Formatted Co2_Oceania"]].head(10))

    with open("top_emissions_in_oceania.txt", "w", encoding='utf-8') as emission_oceania_file:
        emission_oceania_file.write(oceania_emissions[["Country", "Kilotons of Co2", "Formatted Co2_Oceania"]].head(10).to_string(index=False, header=True))
        emission_oceania_file.close()

    top_emitters_oceania = oceania_emissions.head(10)

    plt.figure(figsize=(12, 6))
    plt.bar(top_emitters_oceania["Country"], top_emitters_oceania["Kilotons of Co2"], color="green")
    plt.title("Top 10 CO2 Emitters in Oceaina")
    plt.xlabel("Country")
    plt.ylabel("Total CO2 Emissions (Kilotons)")
    plt.xticks(rotation=45)
    plt.show()

    highest_emitter_oceania = oceania_emissions.iloc[0]
    print(f"The country with the higest CO2 emissions is {highest_emitter_oceania['Country']} with {highest_emitter_oceania['Kilotons of Co2']} kilotons.")


# elif europe_data:
    europe_emissions = europe_data.groupby("Country")["Kilotons of Co2"].sum().reset_index()

    europe_emissions = europe_emissions.sort_values(by="Kilotons of Co2", ascending=False)

    europe_emissions["Formatted Co2_Europe"] = europe_emissions["Kilotons of Co2"].map('{:,.2f}'.format)

    print(europe_emissions[["Country", "Kilotons of Co2", "Formatted Co2_Europe"]].head(10))

    top_emitters_europe = europe_emissions.head(10)

    with open("top_emissions_in_europe.txt", "w", encoding='utf-8') as emission_europe_file:
        emission_europe_file.write(europe_emissions[["Country", "Kilotons of Co2", "Formatted Co2_Europe"]].head(10).to_string(index=False, header=True))
        emission_europe_file.close()


    plt.figure(figsize=(12, 6))
    plt.bar(top_emitters_europe["Country"], top_emitters_europe["Kilotons of Co2"], color="orange")
    plt.title("Top 10 CO2 Emitters in Europe")
    plt.xlabel("Country")
    plt.ylabel("Total CO2 Emissions (Kilotons)")
    plt.xticks(rotation=45)
    plt.show()

    highest_emitter_europe = europe_emissions.iloc[0]
    print(f"The country with the higest CO2 emissions is {highest_emitter_europe['Country']} with {highest_emitter_europe['Kilotons of Co2']} kilotons.")

    break


"""
3) Which countries in Asia have the highest and lowest Metric Tons Per Capita?

"""
# Calculate the average Metric Tons Per Capita by country
avg_metric_tons = asia_data.groupby("Country")["Metric Tons Per Capita"].mean()

# Find the highest and lowest emitters
highest_emitter = avg_metric_tons.idxmax()   # Country with highest
lowest_emitter = avg_metric_tons.idxmin()   # Country with lowest

# Create a DataFrame for visualization
comparison_data = avg_metric_tons[[highest_emitter, lowest_emitter]].reset_index()
comparison_data.columns = ["Country", "Average Metric Tons Per Capita"]

# Bar Chart Visualization
plt.figure(figsize=(8, 6))
plt.bar(comparison_data["Country"], comparison_data["Average Metric Tons Per Capita"], color=["red", "blue"])
plt.title("Countries with Highest and Lowest Metric Tons Per Capita in Asia")
plt.ylabel("Average Metric Tons Per Capita")
plt.xlabel("Country")
plt.xticks(rotation=45)
plt.show()


# Print the hishest and lowest countries
print(f"Highest Metric Tons Per Capita in Asia: {highest_emitter} ({avg_metric_tons[highest_emitter]:.2f})")
print(f"Lowest Metric Tons Per Capita in Asia: {lowest_emitter} ({avg_metric_tons[lowest_emitter]:.2f})")

with open("metric_tones_by_asia.txt", "w", encoding='utf-8') as asia_file:
    asia_file.write(f"Highest Metric Tons Per Capita in Asia: {highest_emitter} ({avg_metric_tons[highest_emitter]:.2f})\n")
    asia_file.write(f"Lowest Metric Tons Per Capita in Asia: {lowest_emitter} ({avg_metric_tons[lowest_emitter]:.2f})")
    asia_file.close()



"""
4) Which countries in all Regions have the highest and lowest Metric Tons Per Capita?

"""
while True:

    # Africa
    avg_metric_tons_africa = africa_data.groupby("Country")["Metric Tons Per Capita"].mean()

    highest_emitter_africa = avg_metric_tons_africa.idxmax()   # Country with highest
    lowest_emitter_africa = avg_metric_tons_africa.idxmin()   # Country with lowest

    comparison_data_africa = avg_metric_tons_africa[[highest_emitter_africa, lowest_emitter_africa]].reset_index()
    comparison_data_africa.columns = ["Country", "Average Metric Tons Per Capita"]

    plt.figure(figsize=(8, 6))
    plt.bar(comparison_data_africa["Country"], comparison_data_africa["Average Metric Tons Per Capita"], color=["red", "blue"])
    plt.title("Countries with Highest and Lowest Metric Tons Per Capita in Africa")
    plt.ylabel("Average Metric Tons Per Capita")
    plt.xlabel("Country")
    plt.xticks(rotation=45)
    plt.show()

    print(f"Highest Metric Tons Per Capita in Africa: {highest_emitter_africa} ({avg_metric_tons_africa[highest_emitter_africa]:.2f})")
    print(f"Lowest Metric Tons Per Capita in Africa: {lowest_emitter_africa} ({avg_metric_tons_africa[lowest_emitter_africa]:.2f})")

    with open("metric_tons_by_africa.txt", "w", encoding='utf-8') as africa_file:
        africa_file.write(f"Highest Metric Tons Per Capita in Africa: {highest_emitter_africa} ({avg_metric_tons_africa[highest_emitter_africa]:.2f})\n")
        africa_file.write(f"Lowest Metric Tons Per Capita in Africa: {lowest_emitter_africa} ({avg_metric_tons_africa[lowest_emitter_africa]:.2f})")
        africa_file.close()
    print()


    # Americas
    avg_metric_tons_americas = americas_data.groupby("Country")["Metric Tons Per Capita"].mean()

    highest_emitter_americas = avg_metric_tons_americas.idxmax()   # Country with highest
    lowest_emitter_americas = avg_metric_tons_americas.idxmin()   # Country with lowest

    comparison_data_americas = avg_metric_tons_americas[[highest_emitter_americas, lowest_emitter_americas]].reset_index()
    comparison_data_americas.columns = ["Country", "Average Metric Tons Per Capita"]

    plt.figure(figsize=(8, 6))
    plt.bar(comparison_data_americas["Country"], comparison_data_americas["Average Metric Tons Per Capita"], color=["red", "blue"])
    plt.title("Countries with Highest and Lowest Metric Tons Per Capita in Americas")
    plt.ylabel("Average Metric Tons Per Capita")
    plt.xlabel("Country")
    plt.xticks(rotation=45)
    plt.show()

    print(f"Highest Metric Tons Per Capita in Americas: {highest_emitter_americas} ({avg_metric_tons_americas[highest_emitter_americas]:.2f})")
    print(f"Lowest Metric Tons Per Capita in Americas: {lowest_emitter_americas} ({avg_metric_tons_americas[lowest_emitter_americas]:.2f})")

    with open("metric_tons_by_americas.txt", "w", encoding='utf-8') as americas_file:
        americas_file.write(f"Highest Metric Tons Per Capita in Americas: {highest_emitter_americas} ({avg_metric_tons_americas[highest_emitter_americas]:.2f})\n")
        americas_file.write(f"Lowest Metric Tons Per Capita in Americas: {lowest_emitter_americas} ({avg_metric_tons_americas[lowest_emitter_americas]:.2f})")
        americas_file.close()
    print()


    # Oceania
    avg_metric_tons_oceania = oceania_data.groupby("Country")["Metric Tons Per Capita"].mean()

    highest_emitter_oceania = avg_metric_tons_oceania.idxmax()   # Country with highest
    lowest_emitter_oceania = avg_metric_tons_oceania.idxmin()   # Country with lowest

    comparison_data_oceania = avg_metric_tons_oceania[[highest_emitter_oceania, lowest_emitter_oceania]].reset_index()
    comparison_data_oceania.columns = ["Country", "Average Metric Tons Per Capita"]

    plt.figure(figsize=(8, 6))
    plt.bar(comparison_data_oceania["Country"], comparison_data_oceania["Average Metric Tons Per Capita"], color=["red", "blue"])
    plt.title("Countries with Highest and Lowest Metric Tons Per Capita in Oceania")
    plt.ylabel("Average Metric Tons Per Capita")
    plt.xlabel("Country")
    plt.xticks(rotation=45)
    plt.show()

    print(f"Highest Metric Tons Per Capita in Oceania: {highest_emitter_oceania} ({avg_metric_tons_oceania[highest_emitter_oceania]:.2f})")
    print(f"Lowest Metric Tons Per Capita in Oceania: {lowest_emitter_oceania} ({avg_metric_tons_oceania[lowest_emitter_oceania]:.2f})")
    
    with open("metric_tons_by_oceania.txt", "w", encoding='utf-8') as oceania_file:
        oceania_file.write(f"Highest Metric Tons Per Capita in Oceania: {highest_emitter_oceania} ({avg_metric_tons_oceania[highest_emitter_oceania]:.2f})\n")
        oceania_file.write(f"Lowest Metric Tons Per Capita in Oceania: {lowest_emitter_oceania} ({avg_metric_tons_oceania[lowest_emitter_oceania]:.2f})")
        oceania_file.close()
    print()

    # Europe
    avg_metric_tons_europe = europe_data.groupby("Country")["Metric Tons Per Capita"].mean()

    highest_emitter_europe = avg_metric_tons_europe.idxmax()   # Country with highest
    lowest_emitter_europe = avg_metric_tons_europe.idxmin()   # Country with lowest

    comparison_data_europe = avg_metric_tons_europe[[highest_emitter_europe, lowest_emitter_europe]].reset_index()
    comparison_data_europe.columns = ["Country", "Average Metric Tons Per Capita"]

    plt.figure(figsize=(8, 6))
    plt.bar(comparison_data_europe["Country"], comparison_data_europe["Average Metric Tons Per Capita"], color=["red", "blue"])
    plt.title("Countries with Highest and Lowest Metric Tons Per Capita in Europe")
    plt.ylabel("Average Metric Tons Per Capita")
    plt.xlabel("Country")
    plt.xticks(rotation=45)
    plt.show()

    print(f"Highest Metric Tons Per Capita in Europe: {highest_emitter_europe} ({avg_metric_tons_europe[highest_emitter_europe]:.2f})")
    print(f"Lowest Metric Tons Per Capita in Europe: {lowest_emitter_europe} ({avg_metric_tons_europe[lowest_emitter_europe]:.2f})")

    with open("metric_tons_by_europe.txt", "w", encoding='utf-8') as europe_file:
        europe_file.write(f"Highest Metric Tons Per Capita in Europe: {highest_emitter_europe} ({avg_metric_tons_europe[highest_emitter_europe]:.2f})\n")
        europe_file.write(f"Lowest Metric Tons Per Capita in Europe: {lowest_emitter_europe} ({avg_metric_tons_europe[lowest_emitter_europe]:.2f})")
        europe_file.close()

    break
