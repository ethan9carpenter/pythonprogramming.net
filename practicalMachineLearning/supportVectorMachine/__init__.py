url = 'https://pythonprogramming.net/support-vector-machine-intro-machine-learning-tutorial/'
description = (
"""
Goal is to derive vector w and constant b in order to create a hyperplane that
separates the two classifications while maximizing the distance that plane
is from the two classifications.

Classification/prediction is done by taking the feature set as vector x and
the predicted class will be the sign of the dot product of x and w plus b.

This value (x-dot-w +b) is know as y

Feature sets that produce a y value of positive or negative one are known as
support vectors because the fall on the margin.  If there are too many of them,
the model may be overfitted and produce poor results.  Because of this, classifiers
usually allow a  certain amount of slack that prevents the model from overfitting
and prevents it from taking too long to run while trying to linearly separate the data.

Linear hyperplanes can always be found by using kernels, functions that manipulate
the featuresets by adding dimensions which creates greater variety and allows
for a separating hyperplane to be found.

Multiple classifications are allowed and classifiers are created though OVO, one-versus-
other, or OVR, one-versus-rest.  These both create multiple hyperplanes separating one
classification from all of the other samples (OVR) or one other specific classification
(OVO) and tests against them when making predictions.
"""
)