
##Final Project Submission 1
##Final_Project_Phase_1.py
##Author: Arin Esterbrook
##Date: 4/9/2022


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##Generate class for all stats calculations as class methods
class Stats_Calculator:
    
    ## Constructor to initalize instance variables of object
    def __init__(self, values, rows, column_name):
        self.values = values
        self.rows = rows
        self.column_name = column_name
    
    #### Definition of methods: ####
    
    ## calculates mean
    def mean(self):
        return round(sum(self.values)/len(self.values),1)
    
    ## calculates median
    def median(self):    
        return round(np.median(self.values), 1)

    ## calculates variance
    def variance(self):
        numerator_total = 0
        for n in self.values:
            numerator_total += (n - self.mean())**2
        return round(numerator_total/ (self.rows - 1), 1)

    ## calculates standard deviation
    def standard_dev(self):
        return round(self.variance()**(1/2), 1)
    
    
    ## prints calculations in desired format
    def __str__(self):
        print(f'Attribute {self.column_name} -----------')
        print(f'Mean: {self.mean():>17}')
        print(f'Median: {self.median():>15}')
        print(f'Variance: {self.variance():>13}')
        print('Standard Deviation:', self.standard_dev())
        return ""


def main():
    
    ##define column names of dataframe
    col =  ["Scn", "A2", "A3", "A4", "A5", "A6", 
            "A7", "A8", "A9", "A10", "Class"]
    
    ## read dataframe
    df = pd.read_csv('breast-cancer-wisconsin.data', 
                     na_values ='?', names = col)
    row, column = df.shape
    
    ## replace NaN values with column A7 mean 
    A7_mean = round(df['A7'].mean(), 1) 
    df = df.fillna(A7_mean)
 
        
    ##create list of Stats_Calculator class instances (objects) for each attribute (column)
    attribute_list = []
    
    ## Append each object to list
    ## create a figure for each attribute (column)
    for c in range(1, len(col)-1):
        values = col[c]
        attribute_list.append(Stats_Calculator(df[values], row, values))
        plt.figure()
        plt.title(f'Histogram of Attribute {values}')
        plt.xlabel('Value of the Attribute')
        plt.ylabel('Number of Data points')
        plt.hist(df[values], bins = 10, color = 'blue', edgecolor = 'black', 
             linewidth = 1.5, alpha= 0.5)

    ## print attribute tables
    for i in range(len(attribute_list)):
        print(attribute_list[i])
        
    
    
    
main()