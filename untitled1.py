
import pandas as pd
import numpy as np
import math
## Phase 3 uses functionality of Phase 2 to configure the centroids
## Initiate class to calculate distance between data points and centroids
## Returns 2 or 4 depending on least distance between centroid and data point 
class Predicted_Class:
    
    def __init__(self, mu_2, mu_4, data_point):
        
        self.data_point = list(data_point[1:10])
        self.calculate_distance(mu_2, mu_4)
        self.cluster = self.predict_cluster()
    ## Calcluate distance between centroids and data point
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
    
    return mu_2, mu_4
## Assigns the data points to centroid cluster 2 or centroid cluster 4
def assign_data_points(df, mu_2, mu_4):
    
    ##Assign each data point to Predicted_class = 2 or Predicted_Class = 4
    predicted_classes = []
    for i in range(len(df)):
        predicted_classes.append(Predicted_Class(mu_2, mu_4, df.loc[i]))
    
    ## Separate clusters by Predicted_class = 2 and Predicted_class = 4
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
    
    ##Create dataframes for Predicted_Class = 2 and Predicted_Class = 4
    cluster_df = pd.DataFrame(cluster_points, columns = cluster_columns)
    row, col = cluster_df.shape
    
    ##Calcluate means for each column
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
        cluster_2_points, cluster_4_points, df = assign_data_points(
                df, mu_2, mu_4)
        new_mu_2 = compute_cluster_means(cluster_2_points)
        new_mu_4 = compute_cluster_means(cluster_4_points)
        
        ## Print results if newly generated centroids are == previous centroid
        if new_mu_2 == mu_2 and new_mu_4 == mu_4:
            final_clusters.append(new_mu_2)
            final_clusters.append(new_mu_4)
            break
        
        ## Continue iteration until centroids do not change (or 50 iterations)
        else:
            mu_2 = new_mu_2
            mu_4 = new_mu_4
            
    return mu_2, mu_4
    
##Phase 3
    
## print the number of data points predicted as class 4, while the correct class is
2
def error_24(df):
    return df.loc[(df['Class'] == 4) & (df['Predicted_Class'] == 2), ['Scn', 
'Class', 'Predicted_Class']]
## print the number of data points predicted as class 2, while the correct class is
4  
def error_42(df):
    return df.loc[(df['Class'] == 2) & (df['Predicted_Class'] == 4), ['Scn', 
'Class', 'Predicted_Class']]
    
## Return the total number of error data points
def error_all(df):
    return error_24(df).shape[0] + error_42(df).shape[0]
## Return dataframe where Predicted_Class is 2
def pclass_2(df):
    return df.loc[df['Predicted_Class'] == 2].shape[0]
## Return dataframe where Predicted_Class is 4
def pclass_4(df):
    return df.loc[df['Predicted_Class'] == 4].shape[0]
        
## Error rate of benign cells (2)
def error_B(df):
    return round((error_24(df).shape[0]/pclass_2(df)) * 100, 1)
   
    
## Error rate of malignant cells (4)
def error_M(df):
    return round((error_42(df).shape[0]/pclass_4(df)) * 100, 1)
## Total error rate
def error_T(df, row):
    return round(((error_all(df)/row) * 100), 1)
## Swap the predicted_class values from 2 to 4 and from 4 to 2
def swap_predicted_class(df):
    df['Predicted_Class'] = np.where(df['Predicted_Class']== 4, 2, 4)
    return df
def main():
    
    ## Define column names of dataframe
    col =  ["Scn", "A2", "A3", "A4", "A5", "A6", 
            "A7", "A8", "A9", "A10", "Class"]
    
    ## Read dataframe
    df = pd.read_csv('breast-cancer-wisconsin.data', 
                     na_values ='?', names = col)
    row, column = df.shape
    
    ## Replace NaN values with column A7 mean 
    A7_mean = round(df['A7'].mean(), 1) 
    df = df.fillna(A7_mean)
 
    
    ## Initialization of centroids Step
    mu_2, mu_4 = initialize_centroids(df)
    ## Assign data points to centroid 2 or centroid 4
    cluster_2_points, cluster_4_points, df = assign_data_points(
            df, mu_2, mu_4)
    
    ## Compute means of each column in clusters
    mu_2 = compute_cluster_means(cluster_2_points)
    mu_4 = compute_cluster_means(cluster_4_points)
    
    ## Iterate assign and recompute steps until the clusters do not change
    ## (equal to the previous centroid)
    recompute_centroids(df, mu_2, mu_4, cluster_2_points, cluster_4_points)
    ## Print total errors
    print(f'Total errors {error_T(df, row)} %')
    
    ## Swap clusters if total error greater than 50%
    if error_T(df, row) >= 50:
        print('Clusters are swapped')
        print('Swapping Predicted_Class')
        swap_predicted_class(df)
        print('\n')
    
    ##print number of data points in the predcited classes attribute    
    print(f'Number of Data points in Predicted Class 2: {pclass_2(df)}')
    print(f'Number of Data points in Predicted Class 4: {pclass_4(df)} \n')
        
    ## print the error_24 data points 
    print('Error data points, Predicted_Class 2: \n')
    print(error_24(df))
    print('\n')
    
    ## print the error_24 data points 
    print('Error data points, Predicted_Class 4: \n')
    print(error_42(df))
    print('\n')
    
    ##print number of all data points
    print(f'Number of all data points: {row} \n')
    
    ##print number of error data points
    print(f'Number of error data points: {error_all(df)} \n')
    
    ##print error rates
    print(f'Error rate for class 2: {error_B(df)} %')
    print(f'Error rate for class 4: {error_M(df)} %')
    print(f'Total error rate: {error_T(df, row)} %')
    
    
main()