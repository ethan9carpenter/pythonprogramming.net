url = 'https://pythonprogramming.net/k-nearest-neighbors-intro-machine-learning-tutorial/'
description = (
    """
    Classifies a featureset by 'voting' among the k featuresets (k-nearest-neighbors)
    that are closest to it in a multidimensional space.  Does so by taking the euclidian
    distance (numpy.linalg.norm) of each sample in the training data and then voting, 
    returning a classificaiton.
    
    Because of the voting-style classification, a confidence score/percentage can be given 
    for each prediction or a set of predicitions.  This score is different from clf.score()
    which tells how accurate the classifier is.
    
    Due to the nature of the classifier, no actual training occurs.  Each prediction involves
    comparing the prediction features to every sample in the training data (though optimization
    is possible).  This causes KNN to be extremely costly in terms of runtime for predictions.
    """
    )