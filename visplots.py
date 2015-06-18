import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from multilayer_perceptron import multilayer_perceptron
import numpy as np

def plotvector(XTrain, yTrain, XTest, yTest, weights, upperLim = 310):
    results = []
    for n in range(1, upperLim, 4):
        clf = KNeighborsClassifier(n_neighbors = n, weights = weights)
        clf = clf.fit(XTrain, yTrain)
        preds = clf.predict(XTest)
        accuracy = clf.score(XTest, yTest)
        results.append([n, accuracy])
    results = np.array(results)
    return(results)

def plotaccuracy(XTrain, yTrain, XTest, yTest, upperLim):
    pltvector1 = plotvector(XTrain, yTrain, XTest, yTest, weights = "uniform",  upperLim=upperLim)
    pltvector2 = plotvector(XTrain, yTrain, XTest, yTest, weights = "distance", upperLim=upperLim)
    line1 = plt.plot(pltvector1[:,0], pltvector1[:,1], label = "uniform",  color='blue')
    line2 = plt.plot(pltvector2[:,0], pltvector2[:,1], label = "distance", color='red')
    plt.legend(loc=3)
    plt.ylim(0.85, 1)
    plt.title('Accuracy with Increasing K')
    plt.xlabel('Number of K nearest neighbors')
    plt.ylabel('Classifiction Accuracy')
    plt.grid()


def knnDecisionPlot(XTrain, yTrain, XTest, yTest, n_neighbors, weights):
    plt.figure(figsize=(7,5))
    h = .02  # step size in the mesh
    Xtrain = XTrain[:, :2] # we only take the first two features.

    # Create color maps
    cmap_light = ListedColormap(["#AAAAFF", "#AAFFAA", "#FFAAAA"])
    cmap_bold  = ListedColormap(["#0000FF", "#00FF00", "#FF0000"])

    clf = KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(Xtrain, yTrain)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = Xtrain[:, 0].min() - 1, Xtrain[:, 0].max() + 1
    y_min, y_max = Xtrain[:, 1].min() - 1, Xtrain[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),  np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    
    plt.pcolormesh(xx, yy, Z, cmap = cmap_light)
    plt.scatter(XTest[:, 0], XTest[:, 1], c = yTest, cmap = cmap_bold)
    plt.contour(xx, yy, Z, colors=['k'], linestyles=['-'], levels=[0])
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel("Radius")
    plt.ylabel("Perimeter")
    plt.title("2-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights))
    plt.show()

def svmDecisionPlot(XTrain, yTrain, XTest, yTest, kernel):
    plt.figure(figsize=(7, 5))

    cmap_light = ListedColormap(["#AAAAFF", "#AAFFAA", "#FFAAAA"])
    cmap_bold = ListedColormap(["#0000FF", "#00FF00", "#FF0000"])
    
    plt.scatter(XTest[:, 0],  XTest[:, 1],  c=yTest,  zorder=10, cmap=cmap_bold)
    plt.axis('tight')

    x_min = XTest[:, 0].min()-2
    x_max = XTest[:, 0].max()+2
    y_min = XTest[:, 1].min()-2
    y_max = XTest[:, 1].max()+2

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]

    if (kernel == 'linear'):
        clf = SVC(kernel='linear')
        clf.fit(XTrain[:,0:2], yTrain)
    else:
        clf = SVC(kernel='rbf', C=1.0, gamma=0.0)
        clf.fit(XTrain[:,0:2], yTrain)

    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=cmap_light)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-.5, 0, .5])
    plt.xlabel("Radius")
    plt.ylabel("Perimeter")
    plt.title(kernel + " SVM")
    plt.show()


def nnDecisionPlot(XTrain, yTrain, XTest, yTest, hidden_layer, learning_rate):
    plt.figure(figsize=(7,5))
    h = .02  # step size in the mesh
    Xtrain = XTrain[:, :2] # we only take the first two features.

    # Create color maps
    cmap_light = ListedColormap(["#AAAAFF", "#AAFFAA", "#FFAAAA"])
    cmap_bold  = ListedColormap(["#0000FF", "#00FF00", "#FF0000"])

    nnet = multilayer_perceptron.MultilayerPerceptronClassifier(activation='logistic',
                                                            hidden_layer_sizes=hidden_layer, learning_rate_init=learning_rate)
    nnet.fit(Xtrain, yTrain)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = Xtrain[:, 0].min() - 1, Xtrain[:, 0].max() + 1
    y_min, y_max = Xtrain[:, 1].min() - 1, Xtrain[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),  np.arange(y_min, y_max, h))
    Z = nnet.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    
    plt.pcolormesh(xx, yy, Z, cmap = cmap_light)
    plt.scatter(XTest[:, 0], XTest[:, 1], c = yTest, cmap = cmap_bold)
    plt.contour(xx, yy, Z, colors=['k'], linestyles=['-'], levels=[0])
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel("Radius")
    plt.ylabel("Perimeter")
    plt.title("2-Class classification (learning rate = %f, hidden layer = '%s')" % (learning_rate, hidden_layer))
    plt.show()

def logregDecisionPlot(XTrain, yTrain, XTest, yTest, pen_val='l2', c_val=10):
    plt.figure(figsize=(7,5))
    h = .02  # step size in the mesh
    Xtrain = XTrain[:, :2] # we only take the first two features.

    # Create color maps
    cmap_light = ListedColormap(["#AAAAFF", "#AAFFAA", "#FFAAAA"])
    cmap_bold  = ListedColormap(["#0000FF", "#00FF00", "#FF0000"])

    l_regression = LogisticRegression(C = c_val, penalty = pen_val)
    l_regression.fit(Xtrain, yTrain)
    

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = Xtrain[:, 0].min() - 1, Xtrain[:, 0].max() + 1
    y_min, y_max = Xtrain[:, 1].min() - 1, Xtrain[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),  np.arange(y_min, y_max, h))
    Z = l_regression.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    
    plt.pcolormesh(xx, yy, Z, cmap = cmap_light)
    plt.scatter(XTest[:, 0], XTest[:, 1], c = yTest, cmap = cmap_bold)
    plt.contour(xx, yy, Z, colors=['k'], linestyles=['-'], levels=[0])
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel("Radius")
    plt.ylabel("Perimeter")
    plt.title("2-Class classification (l2 = '%s', C = '%f')" % (pen_val, c_val))
    plt.show()
