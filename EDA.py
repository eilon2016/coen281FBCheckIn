# Exploratory Data Analysis for the FB Check In Training Set
import numpy
import seaborn
import matplotlib
import matplotlib.pyplot as plt
import pandas
import os

"""
VARIABLE DESCRIPTIONS
Row_id          Unique Row_id           [0, 29,118,020]
x               X coordinate            [0, 10]
y               Y coordinate            [0, 10]
accuracy        Measure of accuracy     [1, 1033]
time            Time of check in        [1, 786239]
place_id        Check in location
"""

def main():
    pandas.set_option('display.float_format', lambda x: '%.3f' % x) # up to 3 decimal places
    matplotlib.interactive(False)
    dataPath = "/Users/Johnny/Documents/FBData/train.csv"
    picklePath = "/Users/Johnny/Documents/FBData/train.pkl"
    # Step 1. Read data set using pandas
    if not os.path.isfile(picklePath):
        data = pandas.read_csv("/Users/Johnny/Documents/FBData/train.csv")
        data.to_pickle(picklePath)
    else:
        data = pandas.read_pickle(picklePath)
    # Using pickle allows for faster subsequent runs

    # print(data.head(10) # print the first 10 rows
    print(data.describe()) # count, mean, std, min, quartiles, max

    # data.boxplot(column='accuracy').plot() # box plot for accuracy
    place_id = data['place_id'] # Place_id is our class/label; it contains a wide range of counts
    #place_id.hist(bins=50).plot()
    place_id_freq = place_id.value_counts()
    # Generates a histogram: x axis is the number of check-ins, y is how many place_ids have that number of check-ins
    print("     place_id_freq")
    print(place_id_freq.describe())
    plt.figure() # Make a new figure
    plt.hist(place_id_freq.values, bins='fd') # Histogram of the counts
    plt.xticks(numpy.arange(0, 1900, 100))
    plt.xlabel("Number of Check-ins")
    plt.ylabel("Number of place_ids")
    plt.savefig('Figures/Check_in_count.png')


if __name__ == "__main__":
    main()
