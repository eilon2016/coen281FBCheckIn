# Exploratory Data Analysis for the FB Check In Training Set
import numpy
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
    #place_id = numpy.log(data['place_id']) # Place_id is our class/label; it contains a wide range of counts
    #place_id.hist(bins=50).plot()
    place_id_freq = data['place_id'].value_counts()
    place_id_freq.hist(bins=100).plot()
    plt.show() # show our plots


if __name__ == "__main__":
    main()
