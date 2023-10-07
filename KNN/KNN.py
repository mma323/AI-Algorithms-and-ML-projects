import pandas as pd
import math


def calculate_distance(feature_vector1, feature_vector2) -> float:
    """
    Calculates the Euclidean distance between two feature vectors.
    """

    feature_differences = []

    for feature1, feature2 in zip(feature_vector1, feature_vector2):
        feature_difference = feature2 - feature1
        squared_feature_difference = math.pow(feature_difference, 2)
        feature_differences.append(squared_feature_difference)
    
    euclidean_distance = math.sqrt( math.fsum(feature_differences) )

    return euclidean_distance


def find_k_nearest_neighbors(
        training_set: pd.DataFrame, 
        test_instance: list, 
        k: int
) -> pd.Series:
    """
    Given a training set, a test instance, and a value for k,
    returns the k nearest neighbors to the test instance,
    based on the Euclidean distance between the test instance
    and each instance in the training set.
    """

    distances = [
        calculate_distance(
            training_set.iloc[i].drop('class'), 
            test_instance
        ) for i in range(len(training_set))
    ]
    
    training_set['Distance'] = distances
    
    k_nearest_neighbors = (
        training_set
            .sort_values(by='Distance')
            .head(k)
            .drop(columns='Distance')
    )

    k_nearest_neighbors = k_nearest_neighbors['class']
    
    return k_nearest_neighbors


def predict_class (
        training_set: pd.DataFrame, 
        test_instance: list,
        k: int
) -> str:
    """
    Given a training set, a test instance, and a value for k,
    returns the predicted class for the test instance,
    based on the k nearest neighbors in the training set.
    """

    k_nearest_neighbors = find_k_nearest_neighbors(
        training_set, 
        test_instance, 
        k 
    )

    prediction = max(
        k_nearest_neighbors, 
        key=k_nearest_neighbors.value_counts().get
    )

    return prediction


def main():
    URL = (
    'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        )
    
    df_iris = pd.read_csv(
        URL,
        header = None,
        names =['sepal length', 
                'sepal width',
                'petal length', 
                'petal width', 
                'class']
    )

    data_point = [7.0, 3.1, 1.3, 0.7]
    print("Given data point: ", data_point)
    print("Prediction:", predict_class(df_iris, data_point, 16) )


if __name__ == "__main__":
    main()