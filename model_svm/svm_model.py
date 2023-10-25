from sklearn.svm import SVC

class SVM_model:
    def __init__(self, x_train, y_train):
        self.X_train = x_train
        self.Y_train = y_train
        self.model = SVC(kernel='linear', probability=True)

    def fit(self):
        self.model.fit(self.X_train, self.Y_train)

    def predict_person(self, image):
        self.fit()
        probabilities = self.model.predict_proba(image)
        return self.model.predict(image), probabilities
