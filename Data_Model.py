import pandas as pd
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


class DataModel:

    def __init__(self, data, target_column="Maintenance_Flag"):
        self.data = data.copy()
        self.target_column = target_column

        
        self.data.columns = self.data.columns.str.strip()

        #removing timestamp if its found
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

    def linreg(self,):
        self.model = LinearRegression()
        self.model.fit(self.X, self.Y)
        return self.model.coef_, self.model.intercept_  # returning the coefficients and intercept of the linear regression model

    
    def split_data(self, train_size=0.7, validation_size=0.15, test_size=0.15):
        n = len(self.data)  # n of rows of my data aka 20k
        train_end = int(train_size * n)
        validation_test_end = int((train_size + validation_size) * n)
        train_data = self.data.iloc[:train_end]
        validation_data = self.data.iloc[train_end:validation_test_end]
        test_data = self.data.iloc[validation_test_end:]
        return train_data, validation_data, test_data

    def MSE(self, y_true, y_pred):
        return ((y_true - y_pred) ** 2).mean()

    def build_nn(self, input_dim, layers=[64, 32], activation="relu", optimizer="adam", loss="mse", batch_size=32):
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
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters)
        self.data["Cluster"] = kmeans.fit_predict(self.X)
        return self.data["Cluster"], kmeans.cluster_centers_

    def gmm(self, n_components=3):
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=n_components)
        self.data["GMM_Cluster"] = gmm.fit_predict(self.X)
        return gmm.means_, self.data["GMM_Cluster"]
       
    
    
    

