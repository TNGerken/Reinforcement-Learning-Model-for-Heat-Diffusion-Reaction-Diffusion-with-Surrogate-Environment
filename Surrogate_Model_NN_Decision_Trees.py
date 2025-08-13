from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load

class SurrogateModel:
    def __init__(self,npz_path,y1_joblib_path, feature_names_txt=None):
        #Loading the neural network weights and parameters
        self.blob= np.load(npz_path, allow_pickle=True)
        self.nL= int(self.blob["n_layers"])
        self.W =[self.blob[f"W{i}"] for i in range(self.nL)]
        self.b =[self.blob[f"b{i}"] for i in range(self.nL)]
        self.actn= str(self.blob["act"])
        
        #Normalization parameters
        self.mu_x, self.sig_x=self.blob["mu_x"],self.blob["sig_x"]
        self.mu_y, self.sig_y= self.blob["mu_y"],self.blob["sig_y"]
        self.feature_names=list(self.blob["feature_names"])
        
        if feature_names_txt and Path(feature_names_txt).exists():
            want = Path(feature_names_txt).read_text(encoding="utf-8").splitlines()
            assert want==self.feature_names, "Feature order mismatch!"
        
        self.y1_model=load(y1_joblib_path)

    def Activation(self, z):
        if self.actn=="relu": 
            return np.maximum(0.0, z)
        elif self.actn== "tanh": 
            return np.tanh(z)
        elif self.actn=="logistic": 
            return 1.0/(1.0+np.exp(-z))
        elif self.actn=="identity": 
            return z
        else: 
            raise ValueError(self.actn)

    def NeuralNetPredict(self, X):
        #Forward pass through the neural network
        Z = (X - self.mu_x)/self.sig_x
        for i in range(self.nL-1):
            Z=self.Activation(Z@self.W[i]+self.b[i])
        Z=Z@self.W[-1]+self.b[-1]
        Y=Z*self.sig_y+self.mu_y
        return Y

    def predict(self, X):
        X=np.asarray(X, dtype=float)
        single= (X.ndim == 1)
        if single: 
            X=X[None, :]
        
        y1=self.y1_model.predict(X).reshape(-1, 1)
        y23=self.NeuralNetPredict(X)
        Y=np.hstack([y1, y23])
        return Y[0] if single else Y

def AddDerivedFeatures(df, nu_O2=3.0/4.0):
    #Adding derived features if they don't exist
    if "lambda" not in df.columns:
        df["lambda"]= df["cO2_in"]/(nu_O2*df["cAl_in"])
    if "excess_O2" not in df.columns:
        df["excess_O2"] =df["cO2_in"]-nu_O2*df["cAl_in"]
    if "excess_Al" not in df.columns:
        df["excess_Al"]=df["cAl_in"]-df["cO2_in"]/nu_O2
    return df

if __name__=="__main__":
    #File paths for model components and data
    NPZ_PATH=r"C:\Users\Trevo\Documents\VisualStudio_Scripts\mlp_surrogate_y23.npz"
    Y1_PATH=r"C:\Users\Trevo\Documents\VisualStudio_Scripts\y1_model.joblib"
    FEATS_PATH=r"C:\Users\Trevo\Documents\VisualStudio_Scripts\feature_names.txt"
    DATA_PATH=r"C:\Users\Trevo\Documents\VisualStudio_Scripts\surrogate_data_u0_Twall_grid.xlsx"

    #Loading surrogate model
    sur=SurrogateModel(NPZ_PATH, Y1_PATH, FEATS_PATH)
    print("Feature order:",sur.feature_names)

    #Loading and preparing the dataset
    df=pd.read_excel(DATA_PATH)
    df.columns=df.columns.str.strip()
    df=AddDerivedFeatures(df)

    #Target variables
    y_cols=["cAl_out","cO2_out","T_out"]

    #Cleaning the data - removing rows with missing values
    need_cols=sur.feature_names + y_cols
    df=df.replace([np.inf,-np.inf], np.nan).dropna(subset=need_cols)

    #Selecting 10 random samples for evaluation
    rng=np.random.default_rng(42)
    idx=rng.choice(len(df), size=10, replace=False)
    df_10=df.iloc[idx].reset_index(drop=True)

    #Preparing inputs and outputs
    X_10=df_10[sur.feature_names].to_numpy(dtype=float)
    y_true=df_10[y_cols].to_numpy(dtype=float)
    y_pred=sur.predict(X_10)

    #Calculating performance metrics
    from sklearn.metrics import mean_squared_error, r2_score
    print("Subset(10) R2:", r2_score(y_true, y_pred))
    for i, name in enumerate(y_cols):
        print(f"{name}: MSE={mean_squared_error(y_true[:,i], y_pred[:,i]):.4e}, R2={r2_score(y_true[:,i], y_pred[:,i]):.4f}")

    #Creating plots for each target variable
    x_axis=np.arange(len(df_10))
    names_pretty=["y1: cAl_out", "y2: cO2_out","y3: T_out"]

    for i, title in enumerate(names_pretty):
        plt.figure()
        plt.plot(x_axis,y_true[:, i],marker="o",linestyle="-",label="Actual")
        plt.plot(x_axis,y_pred[:, i],marker="s",linestyle="--",label="Predicted")
        plt.title(title)
        plt.xlabel("Sample index (random 10)")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
    plt.show()