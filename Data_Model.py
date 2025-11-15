import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression,BayesianRidge 
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.stats import beta, betabinom

class DataModel:

    def __init__(self, data, target_column="Maintenance_Flag"):
        self.data = data.copy()
        self.target_column = target_column

        
        self.data.columns = self.data.columns.str.strip()

        #removing timestamp if its found in the dataset
        if "timestamp" in self.data.columns:
            self.data.drop(columns=["timestamp"], inplace=True)

        #converting machine_status to integers aswell as handeling weird cases.
        if self.target_column.lower() == "machine_status":
            mapping = {"NORMAL": 0, "RECOVERING": 1, "BROKEN": 2}
            self.data[self.target_column] = (
                self.data[self.target_column]
                .astype(str).str.strip().str.upper()
                .map(mapping)
            )

        
        drop_cols = [self.target_column]
        if "Pump_ID" in self.data.columns:
            drop_cols.append("Pump_ID")

        self.X = self.data.drop(columns=drop_cols, errors="ignore").select_dtypes(include=["number"])
        self.Y = self.data[self.target_column]

        #changing NaN's for means to avoid nan errors
        if self.X.isna().any().any():
            self.X = self.X.fillna(self.X.median(numeric_only=True))

        #dropping labels with NaNs
        if self.Y.isna().any():
            mask = self.Y.notna()
            self.X = self.X.loc[mask]
            self.Y = self.Y.loc[mask]

            

        self.X = self.X.dropna(axis=1, how="all")
        self.X = self.X.fillna(self.X.median(numeric_only=True))
        mask = self.X.notna().all(axis=1) & self.Y.notna()
        self.X = self.X.loc[mask]
        self.Y = self.Y.loc[mask]

    def prepare_data(self, dataframe, drop_columns=None):
        """
        Prepare X and Y from a dataframe by dropping specified columns and filling NaNs.
    
        Returns X and Y (features and targets).
        """
        if drop_columns is None:
            drop_columns = [self.target_column]
    
        # Extract features and target
        X = dataframe.drop(columns=drop_columns, errors="ignore").select_dtypes(include=["number"])
        Y = dataframe[self.target_column]
    
        X = X.dropna(axis=1, how="all")

        # Fill NaNs with median
        X = X.fillna(X.median(skipna=True))
    
        # Drop rows where target is NaN
        mask = Y.notna()
        X = X.loc[mask]
        Y = Y.loc[mask]
    
        return X, Y

    def linreg(self,X_train, Y_train):
        """
        Excecutres linear regression on data given.
        
        returns the function's coefefficants, and intercept
        """
        self.model = LinearRegression()
        self.model.fit(X_train, Y_train)
        return self.model.coef_, self.model.intercept_  # returning the coefficients and intercept of the linear regression model

    
    def split_data(self, train_size=0.7, validation_size=0.15, test_size=0.15):
        """
        Use this for preparing data for training and valedating the trainging afterwards.

        returns split data as train_data, Valdidation_data and test_data
        """
        n = len(self.data)  # n of rows of my data aka 20k
        train_end = int(train_size * n)
        validation_test_end = int((train_size + validation_size) * n)
        train_data = self.data.iloc[:train_end]
        validation_data = self.data.iloc[train_end:validation_test_end]
        test_data = self.data.iloc[validation_test_end:]
        return train_data, validation_data, test_data

    def MSE(self, y_true, y_pred):
        """
        computes the mean square error

        returns the MSE of the true y value and the predicted y value
        """
        return ((y_true - y_pred) ** 2).mean()

    def build_nn(self, input_dim, layers=[64, 32], activation="relu", optimizer="adam", loss="mse", batch_size=32):
        """
        builds a neural network with specefied layers
        
        returns a compiled keras model, ready for training
        """
        model = Sequential()
        model.add(Dense(layers[0], input_dim=input_dim, activation=activation))
        for units in layers[1:]:
            model.add(Dense(units, activation=activation))
        model.add(Dense(1)) 
        model.compile(optimizer=optimizer, loss=loss)
        self.nn_model = model
        self.batch_size = batch_size
        return model

    def k_means(self, n_clusters=3):
        """
        Performs K-means clustering on data

        default number of clusters = 3

        returns cluster labels and cluster centers
        """
        kmeans = KMeans(n_clusters=n_clusters)
        self.data["Cluster"] = kmeans.fit_predict(self.X)
        return self.data["Cluster"], kmeans.cluster_centers_

    def gmm(self, n_components=3):
        """
        performs gaussian mixture model clustering on data

        defualt number of gaussian components = 3

        returs cluster labels and cluster means
        
        """
        gmm = GaussianMixture(n_components=n_components)
        self.data["GMM_Cluster"] = gmm.fit_predict(self.X)
        return gmm.means_, self.data["GMM_Cluster"]
    
    def svm_class(self, X_train, Y_train, kernel="rbf", C=1.0, gamma="scale"): 
        # I chose these parametes as they are the default ones accoring to:
        # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

        """
        Training and evaluating an SVM classifier.

        Kernel expects a string
        C expects a float
        gamma expects either scale or auto.

        Returns the trained SVM model.

        """
        # using if, else so that i can use linearSVC for linear and SVC for the others, as LinearSVC is faster
        if kernel == "linear":
            svm_model = make_pipeline(StandardScaler(), LinearSVC(C=C))

        else:
            svm_model = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=C, gamma=gamma))

        
        svm_model.fit(X_train, Y_train)
        self.svm_model = svm_model
        return svm_model
    
    def bayesian_regression(self, X_train, Y_train):
        """
        Computes bayesian regression using bayesianridge from
        sklearn.linear

        returns the trained bayesian model
        """

        bayesian_model = BayesianRidge()
        bayesian_model.fit(X_train, Y_train)

        self.bayesian_model = bayesian_model
        return bayesian_model
    
    def beta_binomial_posterior(self, Y_data, alpha_prior=2, beta_prior=2):
        #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.betabinom.html
        """
        Computes posterior probabilites using the beta-binomial model

        returns a dictonary containing:
        sucsesses
        failures
        alpha posterior
        beta posterior
        posterior mean
        posterior mode
        """
        successes = Y_data.sum()
        faliures = len(Y_data) - successes

        alpha_post = alpha_prior + successes
        beta_post = beta_prior + faliures

        posterior_mean = alpha_post/(alpha_post+beta_post)
        posterior_mode = (alpha_post-1) / (alpha_post+beta_post-2)

        return {"successes" : successes, "failures" : faliures
                ,"alpha_post" : alpha_post, "beta_post" : beta_post
                ,"posterior_mean" : posterior_mean, "posterior_mode" : posterior_mode}



       
    
    
    

