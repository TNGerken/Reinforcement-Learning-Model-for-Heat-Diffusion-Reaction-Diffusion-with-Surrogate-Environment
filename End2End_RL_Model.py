from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from joblib import load

#Safety constants and default values
EPS=1e-8
LENGTH =1.0
ALPHA, BETA =1.0,1.0

def SafeDivision(numer, denom, eps=EPS):
    d = denom if abs(denom) > eps else eps
    return float(numer) / float(d)

class SurrogateModel:
    def __init__(self, npz_path, y1_joblib_path, feature_names_txt=None):
        #Loading the neural network weights and parameters
        self.blob= np.load(npz_path, allow_pickle=True)
        self.nL= int(self.blob["n_layers"])
        self.W =[self.blob[f"W{i}"] for i in range(self.nL)]
        self.b =[self.blob[f"b{i}"] for i in range(self.nL)]
        self.actn=str(self.blob["act"])
        self.mu_x,self.sig_x =self.blob["mu_x"], self.blob["sig_x"]
        self.mu_y,self.sig_y =self.blob["mu_y"], self.blob["sig_y"]
        self.feature_names = list(self.blob["feature_names"])
        
        if feature_names_txt and Path(feature_names_txt).exists():
            want = Path(feature_names_txt).read_text(encoding="utf-8").splitlines()
            assert want == self.feature_names, "Feature order mismatch!"
        
        self.y1_model = load(y1_joblib_path)

    def Activation(self, z):
        if self.actn == "relu": 
            return np.maximum(0.0, z)
        elif self.actn == "tanh": 
            return np.tanh(z)
        elif self.actn == "logistic": 
            return 1.0/(1.0+np.exp(-z))
        elif self.actn=="identity": 
            return z
        else: 
            raise ValueError(self.actn)

    def NeuralNetPredict(self, X):
        Z=(X-self.mu_x)/self.sig_x
        for i in range(self.nL-1):
            Z = self.Activation(Z @ self.W[i] + self.b[i])
        Z=Z@self.W[-1]+self.b[-1]
        Y=Z*self.sig_y+self.mu_y
        return Y

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        single =(X.ndim == 1)
        if single: 
            X = X[None, :]
        
        y1 =self.y1_model.predict(X).reshape(-1, 1)
        y23= self.NeuralNetPredict(X)
        Y=np.hstack([y1, y23])
        return Y[0] if single else Y

def AddChemistryFeatures(d, expect_set, nu_O2_key="nu_O2"):
    #Adding stoichiometry-based features if needed by surrogate
    nu = float(d.get(nu_O2_key,3.0/4.0))
    if nu <= 0.0:
        nu = 3.0/4.0
        d[nu_O2_key] =nu
    
    cAl = max(float(d.get("cAl_in", 100.0)), EPS)
    cO2 = float(d.get("cO2_in", 130.0))
    
    if "lambda" in expect_set and "lambda" not in d: 
        d["lambda"] = SafeDivision(cO2, nu * cAl)
    if "excess_O2" in expect_set and "excess_O2" not in d: 
        d["excess_O2"] = cO2 - nu * cAl
    if "excess_Al" in expect_set and "excess_Al" not in d: 
        d["excess_Al"] = cAl - SafeDivision(cO2, nu)
    return d

def AddPhysicsFeatures(d, expect_set):
    #Computing dimensionless numbers and residence time
    u0=float(d.get("u0", 0.2))
    L=float(d.get("L", LENGTH))
    D=float(d.get("D", 1.0))
    kappa=float(d.get("kappa", 1.0))
    K0=float(d.get("K0",1e3))
    Ea=float(d.get("Ea",1.25e5))
    Ru=float(d.get("Ru",8.314))
    T_in= float(d.get("T_in",1500.0))
    cAl=float(d.get("cAl_in",100.0))
    cO2=float(d.get("cO2_in",130.0))

    if {"tau", "Pe_c", "Pe_T", "Da"} & expect_set:
        tau=L/max(u0, 1e-9)
        if "tau" in expect_set: 
            d["tau"]=tau
        if "Pe_c" in expect_set: 
            d["Pe_c"]=u0*L/max(D, EPS)
        if "Pe_T" in expect_set: 
            d["Pe_T"]=u0*L/max(kappa,EPS)
        if "Da" in expect_set:
            k_ref=K0 * np.exp(-Ea/(Ru*max(T_in, 300.0)))
            d["Da"]=k_ref*(cAl**(ALPHA-1.0))*(cO2**BETA)*tau
    return d

#Configuration parameters
NPZ_PATH=r"C:\Users\Trevo\Documents\VisualStudio_Scripts\mlp_surrogate_y23.npz"
Y1_PATH= r"C:\Users\Trevo\Documents\VisualStudio_Scripts\y1_model.joblib"
FEATS_PATH=r"C:\Users\Trevo\Documents\VisualStudio_Scripts\feature_names.txt"

#Action space definition - only wall temperature and velocity
ACTION_FEATURES=["T_wall", "u0"]
ACTION_LOWS= np.array([1200.0,0.05], dtype=float)
ACTION_HIGHS= np.array([2000.0,0.50], dtype=float)

#Fixed operating conditions
BASE_INPUTS = {
    "T_in": 1500.0, "T_wall": 1600.0,"cAl_in":100.0, "cO2_in": 130.0, "u0": 0.20,
    "D": 1.0,"kappa":1.0, "rho": 1.0, "Cp":1.0, "Qheat": 1.0,
    "nu_O2":3.0/4.0, "u_s": 0.1,"D_Al":1e-3,
    "K0": 1e3,"Ea":1.25e5,"Ru":8.314,"L":LENGTH,
}

#Reward function parameters
TEMP_TARGET=1500.0
W_TEMP, W_cAl,W_cO2=1.0,0.1, 0.1
REWARD_SCALE=1e-6
NOMINAL_ACTION=np.array([0.5,0.5],dtype=float)
OOD_WEIGHT=1e-2

#Training parameters
HORIZON_STEPS=1
SEED=42
LR=5e-4
BATCH_EPISODES=1024
LOGSTD_INIT=-2.0
LOGSTD_MIN=-5.0
ANNEAL_STEP=0.02

class ThermalEnv:
    def __init__(self, surrogate, base_inputs, action_features, action_lows, action_highs,
                 temp_target=TEMP_TARGET, w_temp=W_TEMP, w_cAl=W_cAl, w_cO2=W_cO2,
                 horizon=HORIZON_STEPS, seed=SEED):
        self.sur=surrogate
        self.base_inputs=dict(base_inputs)
        self.action_features=list(action_features)
        self.act_low=np.asarray(action_lows, dtype=float)
        self.act_high=np.asarray(action_highs, dtype=float)
        self.horizon= int(horizon)
        self.rng=np.random.default_rng(seed)

        assert len(self.action_features)==len(self.act_low)==len(self.act_high), "Action dimension mismatch"

        self.expect=set(self.sur.feature_names)
        self.state=None
        self.t=0
        self.temp_target, self.w_temp, self.w_cAl, self.w_cO2=float(temp_target),float(w_temp),float(w_cAl),float(w_cO2)

        self.ValidateInputs()

    def ValidateInputs(self):
        #Checking for missing features and filling with defaults
        derived={"lambda", "excess_O2", "excess_Al", "tau", "Pe_c", "Pe_T", "Da"}
        missing=[f for f in self.sur.feature_names
                   if f not in self.action_features and f not in self.base_inputs and f not in derived]
        
        if missing:
            print("[WARN] Filling unspecified inputs with 0.0 for:", missing)
            for k in missing:
                self.base_inputs[k]=0.0

        extra_actions=[a for a in self.action_features if a not in self.expect]
        if extra_actions:
            raise ValueError(f"Action features not in surrogate: {extra_actions}")

    def ActionToInputs(self,action01):
        #Converting normalized actions to physical values
        phys=self.act_low+action01*(self.act_high-self.act_low)

        d=dict(self.base_inputs)
        for name,v in zip(self.action_features, phys):
            d[name]=float(v)

        AddPhysicsFeatures(d,self.expect)
        AddChemistryFeatures(d,self.expect)

        X=np.array([d[name] for name in self.sur.feature_names], dtype=float)
        return X,d,phys

    def RewardFromAction(self, a01):
        #Computing reward from normalized action
        X,d_in,phys=self.ActionToInputs(a01)
        y=self.sur.predict(X).astype(float)
        cAl_out,cO2_out,T_out=y

        tracking=(T_out-self.temp_target)**2
        leftover=self.w_cAl*(cAl_out**2)+self.w_cO2*(cO2_out**2)
        energy_penalty =0.01 * float(np.sum((phys-self.act_low)**2))
        ood=OOD_WEIGHT*float(np.sum((a01-NOMINAL_ACTION)**2))

        raw_cost= self.w_temp*tracking+leftover+energy_penalty +ood
        reward =-REWARD_SCALE*raw_cost
        return reward,y,phys

    def reset(self):
        self.t= 0
        mid =np.full(len(self.action_features), 0.5, dtype=float)
        r, y, _ =self.RewardFromAction(mid)
        self.state= y.astype(float)
        return self.state.copy()

    def step(self, action01):
        a=np.clip(np.asarray(action01, dtype=float), 0.0, 1.0)
        reward,y,_=self.RewardFromAction(a)
        self.state=y
        self.t+=1
        done=self.t>=self.horizon
        return y.copy(),reward,done

class SimplePolicy:
    def __init__(self,state_dim=3,action_dim=2,hidden_dim=64, lr=LR, seed=SEED):
        self.rng=np.random.default_rng(seed)
        self.state_dim, self.action_dim,self.lr=state_dim,action_dim,lr

        #Network weights initialization
        self.w1=self.rng.normal(0,0.1,size=(state_dim,hidden_dim))
        self.b1=np.zeros(hidden_dim)
        self.w2=self.rng.normal(0,0.1,size=(hidden_dim,hidden_dim))
        self.b2=np.zeros(hidden_dim)
        self.w_mean=self.rng.normal(0,0.1,size=(hidden_dim,action_dim))
        self.b_mean=np.zeros(action_dim)
        self.w_logstd=self.rng.normal(0,0.1,size=(hidden_dim,action_dim))
        self.b_logstd= np.full(action_dim, LOGSTD_INIT)

    @staticmethod
    def Sigmoid(x): 
        return 1.0/(1.0+np.exp(-x))

    def forward(self, s):
        h1= np.maximum(0, s@self.w1 +self.b1)
        h2= np.maximum(0, h1@self.w2+ self.b2)
        mean= self.Sigmoid(h2@self.w_mean+ self.b_mean)
        log_std=h2 @ self.w_logstd+self.b_logstd
        log_std=np.clip(log_std, LOGSTD_MIN, -0.5)
        std=np.exp(log_std)
        return mean, std, (h1, h2)

    def get_action(self,s,training=True):
        mean, std, _ =self.forward(s)
        a= self.rng.normal(mean, std) if training else mean
        return np.clip(a,0.0,1.0), mean, std

    def update(self, states, actions, advantages):
        bs = len(states)
        gw1,gb1=np.zeros_like(self.w1),np.zeros_like(self.b1)
        gw2,gb2=np.zeros_like(self.w2),np.zeros_like(self.b2)
        gwm,gbm=np.zeros_like(self.w_mean),np.zeros_like(self.b_mean)
        gwl,gbl=np.zeros_like(self.w_logstd),np.zeros_like(self.b_logstd)

        for s,a,adv in zip(states,actions,advantages):
            h1p=s @ self.w1+self.b1
            h1=np.maximum(0,h1p)
            h2p=h1 @ self.w2+self.b2
            h2=np.maximum(0,h2p)
            mpre= h2 @ self.w_mean+self.b_mean
            m=1.0/(1.0+np.exp(-mpre))
            lstd=h2@self.w_logstd+self.b_logstd
            lstd=np.clip(lstd,LOGSTD_MIN,-0.5)
            std= np.exp(lstd)

            g_mean = (a-m)/(std**2)
            g_lstd = ((a-m)**2/(std**2)-1.0)
            g_mpre=g_mean*m*(1-m)

            gh2 =(g_mpre @ self.w_mean.T)+(g_lstd @self.w_logstd.T)
            gh2 *=(h2p > 0).astype(float)
            gh1=(gh2@self.w2.T)*(h1p > 0).astype(float)

            gwm+=np.outer(h2,g_mpre)*adv
            gbm+=g_mpre*adv
            gwl+=np.outer(h2,g_lstd)*adv
            gbl+=g_lstd*adv
            gw2+=np.outer(h1,gh2)*adv
            gb2+=gh2*adv
            gw1+=np.outer(s,gh1)*adv
            gb1+=gh1*adv

        self.w_mean +=self.lr*gwm/bs
        self.b_mean +=self.lr*gbm/bs
        self.w_logstd+=self.lr*gwl/bs
        self.b_logstd+=self.lr*gbl/bs
        self.w2+=self.lr*gw2/bs
        self.b2+=self.lr*gb2/bs
        self.w1+=self.lr*gw1/bs
        self.b1+=self.lr*gb1/bs

def SanityScan(env,n=1000):
    #Quick scan of random actions to check reward scale
    rs,Ts,c1s,c2s=[],[],[],[]
    for _ in range(n):
        a01 = np.random.rand(len(env.action_features))
        r, y, _ =env.RewardFromAction(a01)
        cAl_out,cO2_out,T_out=y
        rs.append(r)
        Ts.append(T_out)
        c1s.append(cAl_out)
        c2s.append(cO2_out)
    
    print(f"[scan] reward   min/med/max = {np.min(rs):.3g}/{np.median(rs):.3g}/{np.max(rs):.3g}")
    print(f"[scan] T_out    min/med/max = {np.min(Ts):.3g}/{np.median(Ts):.3g}/{np.max(Ts):.3g}")
    print(f"[scan] cAl_out  min/med/max = {np.min(c1s):.3g}/{np.median(c1s):.3g}/{np.max(c1s):.3g}")
    print(f"[scan] cO2_out  min/med/max = {np.min(c2s):.3g}/{np.median(c2s):.3g}/{np.max(c2s):.3g}")

def MovingAverage(x, k=200):
    if len(x)==0: 
        return np.array([])
    out=[]
    for i in range(len(x)):
        j=max(0,i-k+1)
        out.append(np.mean(x[j:i+1]))
    return np.array(out)

def PlotRewardLandscape(env,nT=60,nU=60):
    #Creating 2D reward contour plot
    t01=np.linspace(0.0,1.0,nT)
    u01=np.linspace(0.0,1.0,nU)

    t_vals=env.act_low[0]+t01*(env.act_high[0]-env.act_low[0])
    u_vals=env.act_low[1]+u01*(env.act_high[1]-env.act_low[1])

    Z = np.empty((nT, nU), dtype=float)
    for i, ti in enumerate(t01):
        for j, uj in enumerate(u01):
            r, _, _=env.RewardFromAction(np.array([ti, uj], dtype=float))
            Z[i,j]=r

    plt.figure(figsize=(8, 6))
    cf = plt.contourf(u_vals,t_vals,Z,levels=40,cmap="viridis")
    plt.xlabel("u0 (m/s)")
    plt.ylabel("T_wall (K)")
    plt.title("Reward landscape (higher is better)")
    plt.colorbar(cf, label="Reward")
    plt.tight_layout()
    plt.show()

def FindBestReward(env,nT=200,nU=200):
    #Grid search for optimal actions
    t01=np.linspace(0.0, 1.0, nT)
    u01=np.linspace(0.0, 1.0, nU)
    best_r,best_t01, best_u01, best_y = -1e9, None, None, None
    
    for ti in t01:
        for uj in u01:
            r, y, _ =env.RewardFromAction(np.array([ti, uj], dtype=float))
            if r > best_r:
                best_r,best_t01,best_u01,best_y = r,ti,uj,y
    
    Tw = env.act_low[0]+best_t01*(env.act_high[0]-env.act_low[0])
    u0 = env.act_low[1]+best_u01*(env.act_high[1]-env.act_low[1])
    print(f"[argmax] Best grid reward = {best_r:.6f} at T_wall ≈ {Tw:.1f} K, u0 ≈ {u0:.3f} m/s | outputs = {best_y}")
    return Tw,u0,best_r,best_y

def TrainPolicy(env,episodes=2000, batch_episodes=BATCH_EPISODES,gamma=1.0):
    #Training the REINFORCE policy
    policy=SimplePolicy(state_dim=3, action_dim=len(env.action_features), lr=LR)
    returns_hist=[]
    best_R, best_a=-np.inf, None

    batch_states, batch_actions, batch_advs = [],[],[]

    for ep in range(episodes):
        s = env.reset()
        a, _, _ =policy.get_action(s, training=True)
        s2, r,done =env.step(a)
        assert done

        returns_hist.append(float(r))
        batch_states.append(s.copy())
        batch_actions.append(a.copy())
        batch_advs.append(float(r))

        if r > best_R:
            best_R, best_a = float(r),a.copy()

        if (ep+1)%batch_episodes == 0:
            adv =np.array(batch_advs, dtype=float)
            adv= (adv -adv.mean())/(adv.std() + 1e-8)
            policy.update(np.array(batch_states), np.array(batch_actions), adv)
            policy.b_logstd = np.maximum(policy.b_logstd - ANNEAL_STEP, LOGSTD_MIN)
            batch_states.clear()
            batch_actions.clear()
            batch_advs.clear()

        if (ep+1)%200==0:
            ma=np.mean(returns_hist[-200:])
            print(f"Ep {ep+1:4d} | MA(200)={ma:.4f} | Best={best_R:.4f} | Best a={best_a}")

    return policy, returns_hist

def EvaluatePolicy(policy, env, episodes=3):
    #Testing the trained policy
    for k in range(episodes):
        s=env.reset()
        a, _, _ = policy.get_action(s, training=False)
        y,r_done,done = env.step(a)
        phys=env.act_low+a*(env.act_high-env.act_low)
        print(f"[Eval {k+1}] Return={r_done:.4f} | action(T_wall,u0)={phys.ravel()} | outputs={y}")

def PlotTrainingCurve(rewards):
    plt.figure(figsize=(11, 5))
    plt.plot(rewards, alpha=0.25, label="Return")
    plt.plot(MovingAverage(rewards, 200), lw=2.5, label="MA(200)")
    plt.title("Training Return")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

#Optional data validation
DO_DATA_CHECK=False
DATA_PATH=r"C:\Users\Trevo\Documents\VisualStudio_Scripts\surrogate_data_wide_grid.xlsx"

def CompareDataPoints(sur):
    #Comparing surrogate predictions with actual data
    import pandas as pd
    from sklearn.metrics import mean_squared_error, r2_score

    df=pd.read_excel(DATA_PATH)
    df.columns=df.columns.str.strip()

    if "lambda" not in df.columns:
        df["lambda"]=df["cO2_in"]/(3.0/4.0*df["cAl_in"])
    if "excess_O2" not in df.columns:
        df["excess_O2"]=df["cO2_in"]-(3.0/4.0)*df["cAl_in"]
    if "excess_Al" not in df.columns:
        df["excess_Al"]=df["cAl_in"]-df["cO2_in"]/(3.0/4.0)

    y_cols=["cAl_out","cO2_out","T_out"]
    need_cols=sur.feature_names + y_cols
    df=df.replace([np.inf, -np.inf], np.nan).dropna(subset=need_cols)

    rng=np.random.default_rng(42)
    idx=rng.choice(len(df), size=10, replace=False)
    df_10= df.iloc[idx].reset_index(drop=True)

    X_10=df_10[sur.feature_names].to_numpy(float)
    y_true=df_10[y_cols].to_numpy(float)
    y_pred=sur.predict(X_10)

    print("Subset(10) R2:", r2_score(y_true, y_pred))
    for i, name in enumerate(y_cols):
        print(f"{name}:MSE={mean_squared_error(y_true[:,i], y_pred[:,i]):.4e},R2={r2_score(y_true[:,i], y_pred[:,i]):.4f}")

    x=np.arange(10)
    names_pretty=["y1: cAl_out","y2: cO2_out","y3: T_out"]
    for i, title in enumerate(names_pretty):
        plt.figure()
        plt.plot(x, y_true[:, i],marker="o",label="Actual")
        plt.plot(x, y_pred[:, i],marker="s",linestyle="--",label="Predicted")
        plt.title(title)
        plt.xlabel("Sample")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
    plt.show()

def main():
    #Loading surrogate model
    sur = SurrogateModel(NPZ_PATH, Y1_PATH, FEATS_PATH)
    print("Surrogate expects",len(sur.feature_names),"features.")
    print(sur.feature_names)

    if DO_DATA_CHECK:
        CompareDataPoints(sur)

    #Creating the environment
    env = ThermalEnv(surrogate=sur, base_inputs=BASE_INPUTS, action_features=ACTION_FEATURES,
                     action_lows=ACTION_LOWS, action_highs=ACTION_HIGHS, temp_target=TEMP_TARGET, 
                     w_temp=W_TEMP, w_cAl=W_cAl, w_cO2=W_cO2, horizon=HORIZON_STEPS, seed=SEED)

    #Quick verification of reward scaling
    SanityScan(env)
    #Visualizing reward landscape and finding optimal point
    PlotRewardLandscape(env,nT=60,nU=60)
    FindBestReward(env, nT=200,nU=200)

    #Training the policy
    policy,rewards =TrainPolicy(env,episodes=2000,batch_episodes=BATCH_EPISODES)

    #Plotting training progress
    PlotTrainingCurve(rewards)
    #Evaluating final policy
    EvaluatePolicy(policy, env, episodes=3)

if __name__=="__main__":
    main()