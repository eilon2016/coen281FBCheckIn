import numpy
import EDA
import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn import metrics


def classification_model(model, data, predictors, outcome):

    fileName = "Results/results_" + str(datetime.datetime.now())
    f = open(fileName + ".txt", 'w')

    # Fit the model:
    model.fit(data[predictors], data[outcome])

    # Make predictions on training set:
    predictions = model.predict(data[predictors])

    # Print accuracy
    accuracy = metrics.accuracy_score(predictions, data[outcome])
    print("Accuracy : %s" % "{0:.3%}".format(accuracy))
    f.write("Accuracy : %s" % "{0:.3%}".format(accuracy))

    # Perform k-fold cross-validation with 5 folds
    kf = KFold(data.shape[0], n_folds=5)
    error = []
    for train, test in kf:
        # Filter training data
        train_predictors = (data[predictors].iloc[train, :])

        # The target we're using to train the algorithm.
        train_target = data[outcome].iloc[train]

        # Training the algorithm using the predictors and target.
        model.fit(train_predictors, train_target)

        # Record error from each cross-validation run
        error.append(model.score(data[predictors].iloc[test, :], data[outcome].iloc[test]))

    print("Cross-Validation Score : %s" % "{0:.3%}".format(numpy.mean(error)))
    f.write("\nCross-Validation Score : %s" % "{0:.3%}".format(numpy.mean(error)))

    # Fit the model again so that it can be referred outside the function:
    model.fit(data[predictors], data[outcome])

    f.close()

def main():
    kNN = KNeighborsClassifier()
    data = EDA.loadData("Data/", "resampled")
    classification_model(model=kNN, data=data, predictors=['x','y'], outcome='place_id')



if __name__ == "__main__":
    main()