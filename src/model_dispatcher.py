from sklearn import tree
from sklearn import ensemble
import xgboost as xgb
import lightgbm as lgbm
import catboost

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


models = {  "dt": tree.DecisionTreeClassifier(criterion="gini", 
                                            max_depth=19, 
                                            min_samples_split=3, 
                                            min_samples_leaf=3
                                            ),

            "random_forest": RandomForestClassifier(
                        n_estimators=100, 
                        criterion="gini", 
                        max_depth=None, 
                        min_samples_split=2, 
                        min_samples_leaf=1, 
                        class_weight="balanced",  # Addressing class imbalance
                        max_features = None,
                        random_state=42
                    ),

            "svm": SVC(
                kernel="linear", 
                class_weight="balanced",  # Addressing class imbalance
                random_state=42
                ),
            
            "xgb": xgb.XGBClassifier(),

            "hist": ensemble.HistGradientBoostingClassifier(), 

            "extra": ensemble.ExtraTreesClassifier(), 

            "lgbm": lgbm.LGBMClassifier(n_jobs=-1),

            'gbm': ensemble.GradientBoostingClassifier(),

            'cat': catboost.CatBoostClassifier(verbose=False),
}
