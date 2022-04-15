from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class PolyLearner:
    def __init__(self, degree):
        self.degree = degree
        self.poly = PolynomialFeatures(degree)
        self.linReg = LinearRegression()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        """Simple string representation for logs and outputs"""
        return ("{}(degree={})").format(
            self.__class__.__name__, self.degree
        )
    
    def fit(self, X, y, sample_weight = None):
        X_Poly = self.poly.fit_transform(X)
        self.linReg.fit(X_Poly, y, sample_weight)
    
    def predict(self, X):
        X_Poly = self.poly.fit_transform(X)
        return self.linReg.predict(X_Poly)