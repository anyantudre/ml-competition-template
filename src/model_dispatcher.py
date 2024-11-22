from sklearn import tree
from sklearn import svm
from sklearn import ensemble
from sklearn import linear_model
from sklearn import neural_network

import xgboost as xgb
import lightgbm as lgbm
import catboost


models = {  
            ########################## linear models ############################
            "logistic": linear_model.LogisticRegression(),
            
            "svm": svm.SVC(random_state=42),

            ########################### tree based models ##########################
            "dt": tree.DecisionTreeClassifier(),

            "random_forest": ensemble.RandomForestClassifier(random_state=42),

            "xgb": xgb.XGBClassifier(n_jobs=-1), # 5 CV: accuracy = 0.91216
            
            "hist": ensemble.HistGradientBoostingClassifier(), 

            "extra": ensemble.ExtraTreesClassifier(), 

            "lgbm": lgbm.LGBMClassifier(n_jobs=-1, verbose=-1),

            'gbm': ensemble.GradientBoostingClassifier(),

            'cat': catboost.CatBoostClassifier(verbose=False),
}
