from justcause.data.generators.ihdp import dgp_on_ihdp

from sklearn.linear_model import LinearRegression
from justcause.learners import SLearner, TLearner, PolyLearner, DragonNet, CausalForest

from justcause.evaluation import evaluate_ite
from justcause.metrics import pehe_score, mean_absolute, enormse, bias

import pandas as pd

from datetime import datetime

settings = ["linear", "sinus", "poly", "mixed", "multi-modal", "expo"]
t_a_settings = ["rct", "single_binary_confounder", "multi_confounder", "single_continuous_confounder"]

learners = [
 
    SLearner(LinearRegression()),
 
    TLearner(LinearRegression()),
    
    SLearner(PolyLearner(2)),
    SLearner(PolyLearner(3)),
 
    TLearner(PolyLearner(2)),
    TLearner(PolyLearner(3)),
    
    DragonNet(),
       
    CausalForest(),
]

for s in settings:
    for t in t_a_settings:
        try:
            f = open("./outputs/info.log", "a")
            timestamp = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
            f.write(f"{timestamp}: Starting setting {s}, {t}\n")
            f.close()

            print(f"{timestamp}: Starting setting {s}, {t}")

            replications = dgp_on_ihdp(setting = s,
                treatment_assignment_setting = t,
                n_samples = 747,
                n_replications = 1000,
                random_state = 0
            )

            results = evaluate_ite(replications, learners, metrics=[pehe_score, mean_absolute, enormse, bias], random_state=0)
            df = pd.DataFrame(results).sort_values('pehe_score-mean')
            df.to_pickle(f"./outputs/{s}_{t}")

        except Exception as e:
            f = open("./outputs/error.log", "a")
            timestamp = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
            f.write(f"{timestamp}: An error occured for {s}, {t}: {e}\n")
            f.close()

            print(f"{timestamp}: An error occured for {s}, {t}: {e}")
    