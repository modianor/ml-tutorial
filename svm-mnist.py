# *_*coding:utf-8 *_*
#
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics

from sklearn import datasets, svm, metrics

# The digits dataset

digits = datasets.load_digits()

images_and_labels = list(zip(digits.images, digits.target))

for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)

    plt.axis('off')

    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')

    plt.title('Training: %i' % label)

# fit model

n_samples = len(digits.images)

data = digits.images.reshape((n_samples, -1))

classifier = svm.SVC(gamma=0.001)

classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

# prediction

expected = digits.target[n_samples // 2:]

predicted = classifier.predict(data[n_samples // 2:])

# print metrics

print("Classification report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(expected, predicted)))

print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))


# accuracy

def accuracy_f():
    m = 0

    correction = 0

    while m < (n_samples // 2):

        if predicted[m] == digits.target[m + n_samples // 2]:
            correction += 1

        m += 1

    return correction


a = accuracy_f()

print(a)

print('the test accuracy is %f :' % (a / (n_samples // 2)))
for index, (image, prediction) in enumerate(images_and_predictions[5:9]):
    plt.subplot(2, 4, index + 5)

    plt.axis('off')

    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')

    plt.title('Prediction: %i' % prediction)

plt.show()
