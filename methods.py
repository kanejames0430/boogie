# Imports
import numpy as np
import plotly.graph_objs as go


def normData(data):
    '''
    z-score normalization of the data as a matrix of floats
    '''
    for att in data:

        # we do not want to normalize PassengerId or Survived Status
        if att != 'PassengerId' and att != 'Survived':
            mean = data[att].mean()
            std = data[att].std()
            z_scores = [(h - mean) / std for h in data[att]]
            data[att] = z_scores
    return data

def condit(i):
    '''
    Quick conditional for lambda function used to recategorize data.
    '''
    if i == "S":
        return 0
    elif i == "Q":
        return 1
    elif i == "C":
        return 2
    
def strToFloat(data):
    '''
    Converts all entries in a dataframe to floats
    '''
    for att in data:
        data[att] = data[att].apply(lambda x: float(x))

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


def cabinToFloat(i):
    cabins = ['C','E','G','D','A','B','F','T']
    if i in cabins:
        return cabins.index(i)
    else:
        return i
    
def matrixBuilder(dict):
    '''
    Creates a matrix from the pandas dataframe of language scores or list of vectors
    '''
    labels = dict["Survived"]
    dict = dict.T
    matrix = []

    for key, vec in dict.items():
        matrix.append(vec)

    return np.array(matrix), labels


def project_to_2d(matrix):
    '''
    Project the matrix onto 2D space using Singular Value Decomposition (SVD).
    '''
    U, _, _ = np.linalg.svd(matrix)
    projected_matrix = U[:, :2]
    return projected_matrix

def project_to_3d(matrix):
    '''
    Project the matrix onto 3D space using Singular Value Decomposition (SVD).
    '''
    U, _, _ = np.linalg.svd(matrix)
    projected_matrix = U[:, :3]
    return projected_matrix

def calculate_distance_from_origin(points):
    '''
    For a list of points, calculate the norm, or the distance from the origin
    '''
    return np.linalg.norm(points, axis=1)

def normalize(array):
    '''
    Normalize the array to a range between 0 and 1.
    '''
    min_val = np.min(array)
    max_val = np.max(array)
    return (array - min_val) / (max_val - min_val)

def plot_2d_projection(matrix, labels):
    # Define colors for the binary labels
    label_colors = {
        0: 'rgb(255, 102, 102)',  # Red for label 0
        1: 'rgb(0, 255, 128)'   # Green for label 1
    }

    # Map the labels to colors
    colors = [label_colors[label[1]] for label in labels]

    trace = go.Scatter(
        x=matrix[:, 0],
        y=matrix[:, 1],
        mode='markers',
        marker=dict(
            size=10,
            color=colors,
            opacity=0.8,
        ),
        text=labels,
        hoverinfo='text'
    )

    layout = go.Layout(
        title='2D Projection',
        xaxis=dict(title='Principal Component 1'),
        yaxis=dict(title='Principal Component 2')
    )

    fig = go.Figure(data=[trace], layout=layout)
    fig.show()

def plot_3d_projection(matrix, labels):
    # Define colors for the binary labels
    label_colors = {
        0: 'rgb(255, 102, 102)',  # Red for label 0
        1: 'rgb(0, 255, 128)'   # Green for label 1
    }

    # Map the labels to colors
    colors = [label_colors[label[1]] for label in labels]

    trace = go.Scatter3d(
        x=matrix[:, 0],
        y=matrix[:, 1],
        z=matrix[:, 2],
        mode='markers',
        marker=dict(
            size=10,
            color=colors,
            opacity=0.8,
        ),
        text=labels,
        hoverinfo='text'
    )

    layout = go.Layout(
        title='3D Projection',
        scene=dict(
            xaxis=dict(title='Principal Component 1'),
            yaxis=dict(title='Principal Component 2'),
            zaxis=dict(title='Principal Component 3')
        )
    )

    fig = go.Figure(data=[trace], layout=layout)
    fig.show()