
##Final Project Submission 2
##Final_Project_Phase_2.py
##Author: Arin Esterbrook
##Date: 4/12/2022


import pandas as pd
import numpy as np
import math


## Initiate class to calculate distance between data points and centroids
## returns 2 or 4 depending on least distance between centroid and data point 
class Predicted_Class:
    
    def __init__(self, mu_2, mu_4, data_point):
        
        self.data_point = list(data_point[1:10])
        self.calculate_distance(mu_2, mu_4)
        self.cluster = self.predict_cluster()


    ## calcluate distance between centroids and data point
    def calculate_distance(self, mu_2, mu_4):
        
        distance_sum_2 = 0
        distance_sum_4 = 0
        
        for n in range(len(self.data_point)):
            distance_sum_2 += (self.data_point[n] - mu_2[n])**2
        self.distance_2 = round(math.sqrt(distance_sum_2), 2)
        
        for n in range(len(self.data_point)):
            distance_sum_4 += (self.data_point[n] - mu_4[n])**2
        self.distance_4 = round(math.sqrt(distance_sum_4), 2)
        
    
    ## Assign 2 or 4 depending on least distance between centroid
    def predict_cluster(self):
         
        if self.distance_4 < self.distance_2:
            return 4
        else:
            return 2
            
## Generates centroids randomly from the df
def initialize_centroids(df):
    
    centroid_index = np.random.randint(len(df), size = (2))
    
    mu_2 = list(df.iloc[centroid_index[0], 1:10])
    mu_4 = list(df.iloc[centroid_index[1], 1:10])      

    
    print(f'Randomly selected row {centroid_index[0]} for centroid mu_2.')
    print(df.iloc[centroid_index[0], 1:10])
    print('\n')
    
    print(f'Randomly selected row {centroid_index[1]} for centroid mu_4.')
    print(df.iloc[centroid_index[1], 1:10])
    print('\n')
    
    return mu_2, mu_4


## Assigns the data points to centroid cluster 2 or centroid cluster 4
def assign_data_points(df, mu_2, mu_4):
    
    ##Assign each data point to Predicted_class = 2 or Predicted_Class = 4
    predicted_classes = []
    for i in range(len(df)):
        predicted_classes.append(Predicted_Class(mu_2, mu_4, df.loc[i]))
    
    ## separate clusters by Predicted_class = 2 and Predicted_class = 4
    cluster_2_points = []
    cluster_4_points = []
    predicted_cluster = []
    for i in range(len(predicted_classes)):
        pc = predicted_classes[i]
        if pc.cluster == 2:
            cluster_2_points.append(pc.data_point)
            predicted_cluster.append(pc.cluster)
        elif pc.cluster == 4:
            cluster_4_points.append(pc.data_point)
            predicted_cluster.append(pc.cluster)
    
    ## Add Predicted_Class column to df      
    df['Predicted_Class'] = predicted_cluster
            
    return cluster_2_points, cluster_4_points, df


## Compute the means from cluster 2 and cluster 4
def compute_cluster_means(cluster_points):
    
    cluster_columns = ["A2", "A3", "A4", "A5", "A6", 
            "A7", "A8", "A9", "A10"]
    
    ##create dataframes for Predicted_Class = 2 and Predicted_Class = 4
    cluster_df = pd.DataFrame(cluster_points, columns = cluster_columns)
    row, col = cluster_df.shape
    
    ##calcluate means for each column
    cluster_means = []
    for c in range(col):
        values = cluster_columns[c]
        cluster_means.append(np.mean(cluster_df[values]))
        
    return cluster_means

## Update centroids based on cluster 2 and cluster 4 means  
def recompute_centroids(df, mu_2, mu_4, cluster_2_points, cluster_4_points):
    
    iterations = 0
    
    while iterations <= 50:
        
        iterations += 1
        final_clusters = []
        cluster_2_points, cluster_4_points, df = assign_data_points(df, mu_2, mu_4)
        new_mu_2 = compute_cluster_means(cluster_2_points)
        new_mu_4 = compute_cluster_means(cluster_4_points)
        
        ## Print results if newly generated centroids are == previous centroid
        
        if new_mu_2 == mu_2 and new_mu_4 == mu_4:
            print(f'Program ended after {iterations} iterations. \n')
            final_clusters.append(new_mu_2)
            final_clusters.append(new_mu_4)
            
     
            ## convert to final_clusters to dataframe 
            cluster_columns = ["A2", "A3", "A4", "A5", "A6", 
            "A7", "A8", "A9", "A10"]
    
            ##create dataframe for final_clusters
            centroid_df = pd.DataFrame(final_clusters, columns = cluster_columns)
            row, col = centroid_df.shape
            
            print(f'Final centroid for mu_2: \n{centroid_df.iloc[0]}')
            print('\n')
            print(f'Final centroid for mu_4: \n{centroid_df.iloc[1]}')
            print('\n')
            print(f'Final cluster assignment: \n')
            print(df[['Scn', 'Class', 'Predicted_Class']].head(21))
           
            break
        
        else:
            mu_2 = new_mu_2
            mu_4 = new_mu_4
            
    return mu_2, mu_4
    

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
 
    
    ## Initialization of centroids Step
    mu_2, mu_4 = initialize_centroids(df)

    ## Assign data points to centroid 2 or centroid 4
    cluster_2_points, cluster_4_points, df = assign_data_points(df, mu_2, mu_4)
    
    ## Compute means of each column in clusters
    mu_2 = compute_cluster_means(cluster_2_points)
    mu_4 = compute_cluster_means(cluster_4_points)
    
    ## Iterate assign and recompute steps until the clusters do not change
    ## (equal to the previous centroid)
    recompute_centroids(df, mu_2, mu_4, cluster_2_points, cluster_4_points)
    
    
    ##final cluster assignment
    cluster_2_points, cluster_4_points, df = assign_data_points(df, mu_2, mu_4)
    
    
main()




