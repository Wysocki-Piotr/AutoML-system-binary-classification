import json

# Pełna lista 78 modeli do klasyfikacji binarnej
models_portfolio =[
    {
        "model_name": "XGBoost_Default",
        "library": "xgboost",
        "class": "XGBClassifier",
        "params": { "n_estimators": 100, "learning_rate": 0.3, "max_depth": 6, "subsample": 1.0, "colsample_bytree": 1.0, "objective": "binary:logistic", "n_jobs": -1, "random_state": 42 }
    },
    {
        "model_name": "XGBoost_Conservative_SmallStep",
        "library": "xgboost",
        "class": "XGBClassifier",
        "params": { "n_estimators": 1000, "learning_rate": 0.01, "max_depth": 4, "subsample": 0.7, "colsample_bytree": 0.7, "reg_alpha": 0.1, "reg_lambda": 1.0, "objective": "binary:logistic", "n_jobs": -1, "random_state": 42 }
    },
    {
        "model_name": "XGBoost_Aggressive_Deep",
        "library": "xgboost",
        "class": "XGBClassifier",
        "params": { "n_estimators": 500, "learning_rate": 0.05, "max_depth": 8, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3, "objective": "binary:logistic", "n_jobs": -1, "random_state": 42 }
    },
    {
        "model_name": "XGBoost_Imbalanced_Weighted",
        "library": "xgboost",
        "class": "XGBClassifier",
        "params": { "n_estimators": 300, "learning_rate": 0.1, "max_depth": 5, "scale_pos_weight": 3.0, "objective": "binary:logistic", "n_jobs": -1, "random_state": 42 }
    },
    {
        "model_name": "XGBoost_Hist_Binning",
        "library": "xgboost",
        "class": "XGBClassifier",
        "params": { "tree_method": "hist", "n_estimators": 200, "learning_rate": 0.1, "max_depth": 6, "objective": "binary:logistic", "n_jobs": -1, "random_state": 42 }
    },
    {
        "model_name": "LightGBM_Default",
        "library": "lightgbm",
        "class": "LGBMClassifier",
        "params": { "n_estimators": 100, "learning_rate": 0.1, "num_leaves": 31, "objective": "binary", "n_jobs": -1, "random_state": 42 }
    },
    {
        "model_name": "LightGBM_LargeLeaves_Dart",
        "library": "lightgbm",
        "class": "LGBMClassifier",
        "params": { "boosting_type": "dart", "n_estimators": 500, "learning_rate": 0.05, "num_leaves": 64, "drop_rate": 0.1, "objective": "binary", "n_jobs": -1, "random_state": 42 }
    },
    {
        "model_name": "LightGBM_ExtraTrees",
        "library": "lightgbm",
        "class": "LGBMClassifier",
        "params": { "extra_trees": True, "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 40, "min_child_samples": 50, "bagging_freq": 1, "bagging_fraction": 0.8, "feature_fraction": 0.8, "objective": "binary", "n_jobs": -1, "random_state": 42 }
    },
    {
        "model_name": "LightGBM_SmallData_Restricted",
        "library": "lightgbm",
        "class": "LGBMClassifier",
        "params": { "n_estimators": 200, "learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 10, "reg_lambda": 10.0, "objective": "binary", "n_jobs": -1, "random_state": 42 }
    },
    {
        "model_name": "LightGBM_GOSS",
        "library": "lightgbm",
        "class": "LGBMClassifier",
        "params": { "boosting_type": "goss", "n_estimators": 200, "learning_rate": 0.1, "num_leaves": 31, "objective": "binary", "n_jobs": -1, "random_state": 42 }
    },
    {
        "model_name": "CatBoost_Default",
        "library": "catboost",
        "class": "CatBoostClassifier",
        "params": { "iterations": 1000, "learning_rate": 0.03, "depth": 6, "verbose": 0, "random_state": 42 }
    },
    {
        "model_name": "CatBoost_Symmetric_Regularized",
        "library": "catboost",
        "class": "CatBoostClassifier",
        "params": { "iterations": 1000, "learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 5, "bagging_temperature": 1, "verbose": 0, "random_state": 42 }
    },
    {
        "model_name": "CatBoost_Depthwise_Growth",
        "library": "catboost",
        "class": "CatBoostClassifier",
        "params": { "grow_policy": "Depthwise", "iterations": 500, "learning_rate": 0.1, "depth": 8, "min_data_in_leaf": 10, "verbose": 0, "random_state": 42 }
    },
    {
        "model_name": "CatBoost_Speed",
        "library": "catboost",
        "class": "CatBoostClassifier",
        "params": { "iterations": 500, "learning_rate": 0.1, "depth": 4, "rsm": 0.8, "verbose": 0, "random_state": 42 }
    },
    {
        "model_name": "TabPFN_Default",
        "library": "tabpfn",
        "class": "TabPFNClassifier",
        "params": { "device": "cpu", "N_ensemble_configurations": 32 }
    },
    {
        "model_name": "HistGradientBoosting_Default",
        "library": "sklearn.ensemble",
        "class": "HistGradientBoostingClassifier",
        "params": { "max_iter": 100, "learning_rate": 0.1, "max_leaf_nodes": 31, "random_state": 42 }
    },
    {
        "model_name": "HistGradientBoosting_Regularized",
        "library": "sklearn.ensemble",
        "class": "HistGradientBoostingClassifier",
        "params": { "max_iter": 500, "learning_rate": 0.05, "max_leaf_nodes": 63, "l2_regularization": 1.0, "early_stopping": True, "random_state": 42 }
    },
    {
        "model_name": "GradientBoosting_Default",
        "library": "sklearn.ensemble",
        "class": "GradientBoostingClassifier",
        "params": { "n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "subsample": 1.0, "random_state": 42 }
    },
    {
        "model_name": "GradientBoosting_Slow_Deep",
        "library": "sklearn.ensemble",
        "class": "GradientBoostingClassifier",
        "params": { "n_estimators": 500, "learning_rate": 0.01, "max_depth": 5, "subsample": 0.7, "random_state": 42 }
    },

    # --- SKLEARN LINEAR ---
    {
        "model_name": "LogisticRegression_L2_Default",
        "library": "sklearn.linear_model",
        "class": "LogisticRegression",
        "params": {"penalty": "l2", "C": 1.0, "solver": "lbfgs", "max_iter": 1000, "n_jobs": -1, "random_state": 42}
    },
    {
        "model_name": "LogisticRegression_L1_Liblinear",
        "library": "sklearn.linear_model",
        "class": "LogisticRegression",
        "params": {"penalty": "l1", "C": 1.0, "solver": "liblinear", "max_iter": 1000, "random_state": 42}
    },
    {
        "model_name": "LogisticRegression_ElasticNet",
        "library": "sklearn.linear_model",
        "class": "LogisticRegression",
        "params": {"penalty": "elasticnet", "C": 0.5, "l1_ratio": 0.5, "solver": "saga", "max_iter": 2000, "n_jobs": -1, "random_state": 42}
    },
    {
        "model_name": "LogisticRegression_StrongReg",
        "library": "sklearn.linear_model",
        "class": "LogisticRegression",
        "params": {"penalty": "l2", "C": 0.01, "solver": "lbfgs", "max_iter": 1000, "n_jobs": -1, "random_state": 42}
    },
    {
        "model_name": "LogisticRegression_WeakReg",
        "library": "sklearn.linear_model",
        "class": "LogisticRegression",
        "params": {"penalty": "l2", "C": 100.0, "solver": "lbfgs", "max_iter": 1000, "n_jobs": -1, "random_state": 42}
    },
    {
        "model_name": "RidgeClassifier_Default",
        "library": "sklearn.linear_model",
        "class": "RidgeClassifier",
        "params": {"alpha": 1.0, "random_state": 42}
    },
    {
        "model_name": "RidgeClassifier_HighAlpha",
        "library": "sklearn.linear_model",
        "class": "RidgeClassifier",
        "params": {"alpha": 10.0, "random_state": 42}
    },
    {
        "model_name": "RidgeClassifier_LowAlpha",
        "library": "sklearn.linear_model",
        "class": "RidgeClassifier",
        "params": {"alpha": 0.1, "random_state": 42}
    },
    {
        "model_name": "SGD_Hinge_SVM",
        "library": "sklearn.linear_model",
        "class": "SGDClassifier",
        "params": {"loss": "hinge", "penalty": "l2", "alpha": 0.0001, "max_iter": 1000, "n_jobs": -1, "random_state": 42}
    },
    {
        "model_name": "SGD_Log_Logistic",
        "library": "sklearn.linear_model",
        "class": "SGDClassifier",
        "params": {"loss": "log_loss", "penalty": "elasticnet", "alpha": 0.001, "l1_ratio": 0.15, "max_iter": 1000, "n_jobs": -1, "random_state": 42}
    },
    {
        "model_name": "SGD_ModifiedHuber",
        "library": "sklearn.linear_model",
        "class": "SGDClassifier",
        "params": {"loss": "modified_huber", "penalty": "l2", "alpha": 0.0001, "max_iter": 1000, "n_jobs": -1, "random_state": 42}
    },
    {
        "model_name": "PassiveAggressive_C1",
        "library": "sklearn.linear_model",
        "class": "PassiveAggressiveClassifier",
        "params": {"C": 1.0, "max_iter": 1000, "tol": 0.001, "random_state": 42}
    },
    {
        "model_name": "PassiveAggressive_LowC",
        "library": "sklearn.linear_model",
        "class": "PassiveAggressiveClassifier",
        "params": {"C": 0.01, "max_iter": 1000, "random_state": 42}
    },
    {
        "model_name": "Perceptron_Default",
        "library": "sklearn.linear_model",
        "class": "Perceptron",
        "params": {"penalty": None, "alpha": 0.0001, "max_iter": 1000, "random_state": 42}
    },
    {
        "model_name": "Perceptron_L2",
        "library": "sklearn.linear_model",
        "class": "Perceptron",
        "params": {"penalty": "l2", "alpha": 0.001, "max_iter": 1000, "random_state": 42}
    },

    # --- SKLEARN SVM ---
    {
        "model_name": "SVC_RBF_Default",
        "library": "sklearn.svm",
        "class": "SVC",
        "params": {"C": 1.0, "kernel": "rbf", "gamma": "scale", "probability": True, "random_state": 42}
    },
    {
        "model_name": "SVC_RBF_Tight",
        "library": "sklearn.svm",
        "class": "SVC",
        "params": {"C": 10.0, "kernel": "rbf", "gamma": "scale", "probability": True, "random_state": 42}
    },
    {
        "model_name": "SVC_Poly_Degree3",
        "library": "sklearn.svm",
        "class": "SVC",
        "params": {"C": 1.0, "kernel": "poly", "degree": 3, "probability": True, "random_state": 42}
    },
    {
        "model_name": "SVC_Linear_Kernel",
        "library": "sklearn.svm",
        "class": "SVC",
        "params": {"C": 1.0, "kernel": "linear", "probability": True, "random_state": 42}
    },
    {
        "model_name": "NuSVC_Default",
        "library": "sklearn.svm",
        "class": "NuSVC",
        "params": {"nu": 0.5, "kernel": "rbf", "gamma": "scale", "probability": True, "random_state": 42}
    },
    {
        "model_name": "NuSVC_Loose",
        "library": "sklearn.svm",
        "class": "NuSVC",
        "params": {"nu": 0.1, "kernel": "rbf", "probability": True, "random_state": 42}
    },
    {
        "model_name": "LinearSVC_Default",
        "library": "sklearn.svm",
        "class": "LinearSVC",
        "params": {"penalty": "l2", "loss": "squared_hinge", "C": 1.0, "dual": False, "max_iter": 2000, "random_state": 42}
    },
    {
        "model_name": "LinearSVC_L1_Primal",
        "library": "sklearn.svm",
        "class": "LinearSVC",
        "params": {"penalty": "l1", "loss": "squared_hinge", "C": 0.5, "dual": False, "max_iter": 2000, "random_state": 42}
    },
    {
        "model_name": "LinearSVC_Hinge",
        "library": "sklearn.svm",
        "class": "LinearSVC",
        "params": {"penalty": "l2", "loss": "hinge", "C": 1.0, "dual": True, "max_iter": 2000, "random_state": 42}
    },

    # --- SKLEARN NEIGHBORS ---
    {
        "model_name": "KNN_5_Uniform",
        "library": "sklearn.neighbors",
        "class": "KNeighborsClassifier",
        "params": {"n_neighbors": 5, "weights": "uniform", "metric": "minkowski", "n_jobs": -1}
    },
    {
        "model_name": "KNN_15_Distance",
        "library": "sklearn.neighbors",
        "class": "KNeighborsClassifier",
        "params": {"n_neighbors": 15, "weights": "distance", "metric": "minkowski", "n_jobs": -1}
    },
    {
        "model_name": "KNN_5_Manhattan",
        "library": "sklearn.neighbors",
        "class": "KNeighborsClassifier",
        "params": {"n_neighbors": 5, "weights": "uniform", "metric": "manhattan", "n_jobs": -1}
    },
    {
        "model_name": "NearestCentroid_Euclidean",
        "library": "sklearn.neighbors",
        "class": "NearestCentroid",
        "params": {"metric": "euclidean"}
    },
    {
        "model_name": "NearestCentroid_Manhattan",
        "library": "sklearn.neighbors",
        "class": "NearestCentroid",
        "params": {"metric": "manhattan"}
    },

    # --- SKLEARN NAIVE BAYES ---
    {
        "model_name": "GaussianNB_Default",
        "library": "sklearn.naive_bayes",
        "class": "GaussianNB",
        "params": {"var_smoothing": 1e-09}
    },
    {
        "model_name": "GaussianNB_Smoothed",
        "library": "sklearn.naive_bayes",
        "class": "GaussianNB",
        "params": {"var_smoothing": 1e-05}
    },
    {
        "model_name": "BernoulliNB_Default",
        "library": "sklearn.naive_bayes",
        "class": "BernoulliNB",
        "params": {"alpha": 1.0, "binarize": 0.0}
    },
    {
        "model_name": "MultinomialNB_Default",
        "library": "sklearn.naive_bayes",
        "class": "MultinomialNB",
        "params": {"alpha": 1.0}
    },
    {
        "model_name": "ComplementNB_Default",
        "library": "sklearn.naive_bayes",
        "class": "ComplementNB",
        "params": {"alpha": 1.0}
    },

    # --- SKLEARN DISCRIMINANT ANALYSIS ---
    {
        "model_name": "LDA_SVD",
        "library": "sklearn.discriminant_analysis",
        "class": "LinearDiscriminantAnalysis",
        "params": {"solver": "svd"}
    },
    {
        "model_name": "LDA_LSQR_Shrinkage",
        "library": "sklearn.discriminant_analysis",
        "class": "LinearDiscriminantAnalysis",
        "params": {"solver": "lsqr", "shrinkage": "auto"}
    },
    {
        "model_name": "QDA_Default",
        "library": "sklearn.discriminant_analysis",
        "class": "QuadraticDiscriminantAnalysis",
        "params": {"reg_param": 0.0}
    },
    {
        "model_name": "QDA_Regularized",
        "library": "sklearn.discriminant_analysis",
        "class": "QuadraticDiscriminantAnalysis",
        "params": {"reg_param": 0.1}
    },

    # --- SKLEARN NEURAL NETWORKS ---
    {
        "model_name": "MLP_Default_100",
        "library": "sklearn.neural_network",
        "class": "MLPClassifier",
        "params": {"hidden_layer_sizes": (100, 50), "activation": "relu", "solver": "adam", "alpha": 0.0001, "max_iter": 500, "random_state": 42}
    },
    {
        "model_name": "MLP_Layer100_50",
        "library": "sklearn.neural_network",
        "class": "MLPClassifier",
        "params": {"hidden_layer_sizes": (100, 50), "activation": "relu", "solver": "adam", "alpha": 0.0001, "max_iter": 500, "random_state": 42}
    },
    {
        "model_name": "MLP_Deep_Reg",
        "library": "sklearn.neural_network",
        "class": "MLPClassifier",
        "params": {"hidden_layer_sizes": (100, 50), "activation": "relu", "solver": "adam", "alpha": 0.01, "max_iter": 500, "random_state": 42}
    },
    {
        "model_name": "MLP_Wide_Tanh",
        "library": "sklearn.neural_network",
        "class": "MLPClassifier",
        "params": {"hidden_layer_sizes": (100, 50), "activation": "tanh", "solver": "adam", "alpha": 0.001, "max_iter": 500, "random_state": 42}
    },
    {
        "model_name": "MLP_LBFGS_SmallData",
        "library": "sklearn.neural_network",
        "class": "MLPClassifier",
        "params": {"hidden_layer_sizes": (100, 50), "activation": "relu", "solver": "lbfgs", "alpha": 0.0001, "max_iter": 1000, "random_state": 42}
    },

    # --- SKLEARN TREES (SINGLE) ---
    {
        "model_name": "DecisionTree_Default",
        "library": "sklearn.tree",
        "class": "DecisionTreeClassifier",
        "params": {"criterion": "gini", "max_depth": None, "random_state": 42}
    },
    {
        "model_name": "DecisionTree_MaxDepth5",
        "library": "sklearn.tree",
        "class": "DecisionTreeClassifier",
        "params": {"criterion": "gini", "max_depth": 5, "min_samples_split": 5, "random_state": 42}
    },
    {
        "model_name": "DecisionTree_Entropy_Reg",
        "library": "sklearn.tree",
        "class": "DecisionTreeClassifier",
        "params": {"criterion": "entropy", "max_depth": 10, "min_samples_leaf": 10, "random_state": 42}
    },
    {
        "model_name": "ExtraTree_Default",
        "library": "sklearn.tree",
        "class": "ExtraTreeClassifier",
        "params": {"criterion": "gini", "random_state": 42}
    },

    # --- OTHERS / SPECIAL ---
    {
        "model_name": "GaussianProcess_RBF",
        "library": "sklearn.gaussian_process",
        "class": "GaussianProcessClassifier",
        "params": {"kernel": "1.0 * RBF(1.0)", "random_state": 42, "n_jobs": -1}
    },
    {
        "model_name": "Dummy_MostFrequent",
        "library": "sklearn.dummy",
        "class": "DummyClassifier",
        "params": {"strategy": "most_frequent", "random_state": 42}
    },
    {
        "model_name": "TabPFN_Default",
        "library": "tabpfn",
        "class": "TabPFNClassifier",
        "params": {"device": "cpu", "N_ensemble_configurations": 32}
    },
    ################################
    {
    "model_name": "LogisticRegression_L2",
    "library": "sklearn.linear_model",
    "class": "LogisticRegression",
    "params": { "penalty": "l2", "C": 1.0, "solver": "lbfgs", "max_iter": 1000, "n_jobs": -1, "random_state": 42 }
  },
  {
    "model_name": "LogisticRegression_L1_Liblinear",
    "library": "sklearn.linear_model",
    "class": "LogisticRegression",
    "params": { "penalty": "l1", "C": 1.0, "solver": "liblinear", "max_iter": 1000, "random_state": 42 }
  },
  {
    "model_name": "LogisticRegression_ElasticNet",
    "library": "sklearn.linear_model",
    "class": "LogisticRegression",
    "params": { "penalty": "elasticnet", "C": 0.5, "l1_ratio": 0.5, "solver": "saga", "max_iter": 2000, "n_jobs": -1, "random_state": 42 }
  },
  {
    "model_name": "LogisticRegression_StrongReg",
    "library": "sklearn.linear_model",
    "class": "LogisticRegression",
    "params": { "penalty": "l2", "C": 0.01, "solver": "lbfgs", "max_iter": 1000, "n_jobs": -1, "random_state": 42 }
  },
  {
    "model_name": "LogisticRegression_WeakReg",
    "library": "sklearn.linear_model",
    "class": "LogisticRegression",
    "params": { "penalty": "l2", "C": 100.0, "solver": "lbfgs", "max_iter": 1000, "n_jobs": -1, "random_state": 42 }
  },
  {
    "model_name": "RidgeClassifier_Default",
    "library": "sklearn.linear_model",
    "class": "RidgeClassifier",
    "params": { "alpha": 1.0, "random_state": 42 }
  },
  {
    "model_name": "RidgeClassifier_HighAlpha",
    "library": "sklearn.linear_model",
    "class": "RidgeClassifier",
    "params": { "alpha": 10.0, "random_state": 42 }
  },
  {
    "model_name": "SGD_Hinge_SVM",
    "library": "sklearn.linear_model",
    "class": "SGDClassifier",
    "params": { "loss": "hinge", "penalty": "l2", "alpha": 0.0001, "max_iter": 1000, "n_jobs": -1, "random_state": 42 }
  },
  {
    "model_name": "SGD_Log_Logistic",
    "library": "sklearn.linear_model",
    "class": "SGDClassifier",
    "params": { "loss": "log_loss", "penalty": "elasticnet", "alpha": 0.001, "l1_ratio": 0.15, "max_iter": 1000, "n_jobs": -1, "random_state": 42 }
  },
  {
    "model_name": "SGD_ModifiedHuber",
    "library": "sklearn.linear_model",
    "class": "SGDClassifier",
    "params": { "loss": "modified_huber", "penalty": "l2", "alpha": 0.0001, "max_iter": 1000, "n_jobs": -1, "random_state": 42 }
  },
  {
    "model_name": "PassiveAggressive_C1",
    "library": "sklearn.linear_model",
    "class": "PassiveAggressiveClassifier",
    "params": { "C": 1.0, "max_iter": 1000, "tol": 0.001, "random_state": 42 }
  },
  {
    "model_name": "Perceptron_Default",
    "library": "sklearn.linear_model",
    "class": "Perceptron",
    "params": { "penalty": None, "alpha": 0.0001, "max_iter": 1000, "random_state": 42 }
  },
  {
    "model_name": "LinearSVC_Default",
    "library": "sklearn.svm",
    "class": "LinearSVC",
    "params": { "penalty": "l2", "loss": "squared_hinge", "C": 1.0, "dual": False, "max_iter": 2000, "random_state": 42 }
  },
  {
    "model_name": "LinearSVC_L1_Primal",
    "library": "sklearn.svm",
    "class": "LinearSVC",
    "params": { "penalty": "l1", "loss": "squared_hinge", "C": 0.5, "dual": False, "max_iter": 2000, "random_state": 42 }
  },
  {
    "model_name": "SVC_RBF_Default",
    "library": "sklearn.svm",
    "class": "SVC",
    "params": { "C": 1.0, "kernel": "rbf", "gamma": "scale", "probability": True, "random_state": 42 }
  },
  {
    "model_name": "SVC_RBF_Tight",
    "library": "sklearn.svm",
    "class": "SVC",
    "params": { "C": 10.0, "kernel": "rbf", "gamma": "scale", "probability": True, "random_state": 42 }
  },
  {
    "model_name": "SVC_Poly_Degree3",
    "library": "sklearn.svm",
    "class": "SVC",
    "params": { "C": 1.0, "kernel": "poly", "degree": 3, "probability": True, "random_state": 42 }
  },
  {
    "model_name": "NuSVC_Default",
    "library": "sklearn.svm",
    "class": "NuSVC",
    "params": { "nu": 0.5, "kernel": "rbf", "gamma": "scale", "probability": True, "random_state": 42 }
  },
  {
    "model_name": "KNN_5_Uniform",
    "library": "sklearn.neighbors",
    "class": "KNeighborsClassifier",
    "params": { "n_neighbors": 5, "weights": "uniform", "metric": "minkowski", "n_jobs": -1 }
  },
  {
    "model_name": "KNN_15_Distance",
    "library": "sklearn.neighbors",
    "class": "KNeighborsClassifier",
    "params": { "n_neighbors": 15, "weights": "distance", "metric": "minkowski", "n_jobs": -1 }
  },
  {
    "model_name": "KNN_5_Manhattan",
    "library": "sklearn.neighbors",
    "class": "KNeighborsClassifier",
    "params": { "n_neighbors": 5, "weights": "uniform", "metric": "manhattan", "n_jobs": -1 }
  },
  {
    "model_name": "NearestCentroid",
    "library": "sklearn.neighbors",
    "class": "NearestCentroid",
    "params": { "metric": "euclidean" }
  },
  {
    "model_name": "GaussianNB_Default",
    "library": "sklearn.naive_bayes",
    "class": "GaussianNB",
    "params": { "var_smoothing": 1e-09 }
  },
  {
    "model_name": "BernoulliNB_Default",
    "library": "sklearn.naive_bayes",
    "class": "BernoulliNB",
    "params": { "alpha": 1.0, "binarize": 0.0 }
  },
  {
    "model_name": "LDA_SVD",
    "library": "sklearn.discriminant_analysis",
    "class": "LinearDiscriminantAnalysis",
    "params": { "solver": "svd" }
  },
  {
    "model_name": "QDA_Default",
    "library": "sklearn.discriminant_analysis",
    "class": "QuadraticDiscriminantAnalysis",
    "params": { "reg_param": 0.0 }
  },
  {
    "model_name": "GaussianProcess_RBF",
    "library": "sklearn.gaussian_process",
    "class": "GaussianProcessClassifier",
    "params": { "kernel": "1.0 * RBF(1.0)", "random_state": 42, "n_jobs": -1 }
  },
  {
    "model_name": "Dummy_MostFrequent",
    "library": "sklearn.dummy",
    "class": "DummyClassifier",
    "params": { "strategy": "most_frequent", "random_state": 42 }
  },
  {
    "model_name": "XGBoost_Default",
    "library": "xgboost",
    "class": "XGBClassifier",
    "params": { "n_estimators": 100, "learning_rate": 0.3, "max_depth": 6, "subsample": 1.0, "colsample_bytree": 1.0, "objective": "binary:logistic", "n_jobs": -1, "random_state": 42 }
  },
  {
    "model_name": "XGBoost_Conservative_SmallStep",
    "library": "xgboost",
    "class": "XGBClassifier",
    "params": { "n_estimators": 1000, "learning_rate": 0.01, "max_depth": 4, "subsample": 0.7, "colsample_bytree": 0.7, "reg_alpha": 0.1, "reg_lambda": 1.0, "objective": "binary:logistic", "n_jobs": -1, "random_state": 42 }
  },
  {
    "model_name": "XGBoost_Aggressive_Deep",
    "library": "xgboost",
    "class": "XGBClassifier",
    "params": { "n_estimators": 500, "learning_rate": 0.05, "max_depth": 8, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3, "objective": "binary:logistic", "n_jobs": -1, "random_state": 42 }
  },
  {
    "model_name": "XGBoost_Imbalanced_Weighted",
    "library": "xgboost",
    "class": "XGBClassifier",
    "params": { "n_estimators": 300, "learning_rate": 0.1, "max_depth": 5, "scale_pos_weight": 3.0, "objective": "binary:logistic", "n_jobs": -1, "random_state": 42 }
  },
  {
    "model_name": "XGBoost_Hist_Binning",
    "library": "xgboost",
    "class": "XGBClassifier",
    "params": { "tree_method": "hist", "n_estimators": 200, "learning_rate": 0.1, "max_depth": 6, "objective": "binary:logistic", "n_jobs": -1, "random_state": 42 }
  },
  {
    "model_name": "LightGBM_Default",
    "library": "lightgbm",
    "class": "LGBMClassifier",
    "params": { "n_estimators": 100, "learning_rate": 0.1, "num_leaves": 31, "objective": "binary", "n_jobs": -1, "random_state": 42 }
  },
  {
    "model_name": "LightGBM_LargeLeaves_Dart",
    "library": "lightgbm",
    "class": "LGBMClassifier",
    "params": { "boosting_type": "dart", "n_estimators": 500, "learning_rate": 0.05, "num_leaves": 64, "drop_rate": 0.1, "objective": "binary", "n_jobs": -1, "random_state": 42 }
  },
  {
    "model_name": "LightGBM_ExtraTrees",
    "library": "lightgbm",
    "class": "LGBMClassifier",
    "params": { "extra_trees": True, "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 40, "min_child_samples": 50, "bagging_freq": 1, "bagging_fraction": 0.8, "feature_fraction": 0.8, "objective": "binary", "n_jobs": -1, "random_state": 42 }
  },
  {
    "model_name": "LightGBM_SmallData_Restricted",
    "library": "lightgbm",
    "class": "LGBMClassifier",
    "params": { "n_estimators": 200, "learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 10, "reg_lambda": 10.0, "objective": "binary", "n_jobs": -1, "random_state": 42 }
  },
  {
    "model_name": "LightGBM_GOSS",
    "library": "lightgbm",
    "class": "LGBMClassifier",
    "params": { "boosting_type": "goss", "n_estimators": 200, "learning_rate": 0.1, "num_leaves": 31, "objective": "binary", "n_jobs": -1, "random_state": 42 }
  },
  {
    "model_name": "CatBoost_Default",
    "library": "catboost",
    "class": "CatBoostClassifier",
    "params": { "iterations": 1000, "learning_rate": 0.03, "depth": 6, "verbose": 0, "random_state": 42 }
  },
  {
    "model_name": "CatBoost_Symmetric_Regularized",
    "library": "catboost",
    "class": "CatBoostClassifier",
    "params": { "iterations": 1000, "learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 5, "bagging_temperature": 1, "verbose": 0, "random_state": 42 }
  },
  {
    "model_name": "CatBoost_Depthwise_Growth",
    "library": "catboost",
    "class": "CatBoostClassifier",
    "params": { "grow_policy": "Depthwise", "iterations": 500, "learning_rate": 0.1, "depth": 8, "min_data_in_leaf": 10, "verbose": 0, "random_state": 42 }
  },
  {
    "model_name": "CatBoost_Speed",
    "library": "catboost",
    "class": "CatBoostClassifier",
    "params": { "iterations": 500, "learning_rate": 0.1, "depth": 4, "rsm": 0.8, "verbose": 0, "random_state": 42 }
  },
  {
    "model_name": "TabPFN_Default",
    "library": "tabpfn",
    "class": "TabPFNClassifier",
    "params": { "device": "cpu", "N_ensemble_configurations": 32 }
  },
  {
    "model_name": "HistGradientBoosting_Default",
    "library": "sklearn.ensemble",
    "class": "HistGradientBoostingClassifier",
    "params": { "max_iter": 100, "learning_rate": 0.1, "max_leaf_nodes": 31, "random_state": 42 }
  },
  {
    "model_name": "HistGradientBoosting_Regularized",
    "library": "sklearn.ensemble",
    "class": "HistGradientBoostingClassifier",
    "params": { "max_iter": 500, "learning_rate": 0.05, "max_leaf_nodes": 63, "l2_regularization": 1.0, "early_stopping": True, "random_state": 42 }
  },
  {
    "model_name": "GradientBoosting_Default",
    "library": "sklearn.ensemble",
    "class": "GradientBoostingClassifier",
    "params": { "n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "subsample": 1.0, "random_state": 42 }
  },
  {
    "model_name": "GradientBoosting_Slow_Deep",
    "library": "sklearn.ensemble",
    "class": "GradientBoostingClassifier",
    "params": { "n_estimators": 500, "learning_rate": 0.01, "max_depth": 5, "subsample": 0.7, "random_state": 42 }
  },{
    "model_name": "KNN_15_Distance",
    "library": "sklearn.neighbors",
    "class": "KNeighborsClassifier",
    "params": { "n_neighbors": 15, "weights": "distance", "metric": "minkowski", "n_jobs": -1 }
  },
  {
    "model_name": "KNN_5_Manhattan",
    "library": "sklearn.neighbors",
    "class": "KNeighborsClassifier",
    "params": { "n_neighbors": 5, "weights": "uniform", "metric": "manhattan", "n_jobs": -1 }
  },
  {
    "model_name": "BernoulliNB_Default",
    "library": "sklearn.naive_bayes",
    "class": "BernoulliNB",
    "params": { "alpha": 1.0, "binarize": 0.0 }
  },
  {
    "model_name": "GaussianNB_Default",
    "library": "sklearn.naive_bayes",
    "class": "GaussianNB",
    "params": { "var_smoothing": 1e-09 }
  },
  {
    "model_name": "MultinomialNB_Default",
    "library": "sklearn.naive_bayes",
    "class": "MultinomialNB",
    "params": { "alpha": 1.0 }
  },
  {
    "model_name": "ComplementNB_Default",
    "library": "sklearn.naive_bayes",
    "class": "ComplementNB",
    "params": { "alpha": 1.0 }
  },
  {
    "model_name": "DecisionTree_Default",
    "library": "sklearn.tree",
    "class": "DecisionTreeClassifier",
    "params": { "criterion": "gini", "max_depth": None, "random_state": 42 }
  },
  {
    "model_name": "DecisionTree_MaxDepth5",
    "library": "sklearn.tree",
    "class": "DecisionTreeClassifier",
    "params": { "criterion": "gini", "max_depth": 5, "min_samples_split": 5, "random_state": 42 }
  },
  {
    "model_name": "DecisionTree_Entropy_Regularized",
    "library": "sklearn.tree",
    "class": "DecisionTreeClassifier",
    "params": { "criterion": "entropy", "max_depth": 10, "min_samples_leaf": 10, "random_state": 42 }
  },
  {
    "model_name": "PassiveAggressive_C1",
    "library": "sklearn.linear_model",
    "class": "PassiveAggressiveClassifier",
    "params": { "C": 1.0, "max_iter": 1000, "tol": 0.001, "random_state": 42 }
  },
  {
    "model_name": "PassiveAggressive_LowC",
    "library": "sklearn.linear_model",
    "class": "PassiveAggressiveClassifier",
    "params": { "C": 0.01, "max_iter": 1000, "random_state": 42 }
  },
  {
    "model_name": "LDA_SVD",
    "library": "sklearn.discriminant_analysis",
    "class": "LinearDiscriminantAnalysis",
    "params": { "solver": "svd" }
  },
  {
    "model_name": "LDA_LSQR_Shrinkage",
    "library": "sklearn.discriminant_analysis",
    "class": "LinearDiscriminantAnalysis",
    "params": { "solver": "lsqr", "shrinkage": "auto" }
  },
  {
    "model_name": "QDA_Default",
    "library": "sklearn.discriminant_analysis",
    "class": "QuadraticDiscriminantAnalysis",
    "params": { "reg_param": 0.0 }
  },
  {
    "model_name": "QDA_Regularized",
    "library": "sklearn.discriminant_analysis",
    "class": "QuadraticDiscriminantAnalysis",
    "params": { "reg_param": 0.1 }
  },
  {
    "model_name": "NearestCentroid_Euclidean",
    "library": "sklearn.neighbors",
    "class": "NearestCentroid",
    "params": { "metric": "euclidean" }
  },
  {
    "model_name": "Perceptron_Default",
    "library": "sklearn.linear_model",
    "class": "Perceptron",
    "params": { "penalty": None, "alpha": 0.0001, "max_iter": 1000, "random_state": 42 }
  },
  {
    "model_name": "GaussianProcess_RBF",
    "library": "sklearn.gaussian_process",
    "class": "GaussianProcessClassifier",
    "params": { "kernel": "1.0 * RBF(1.0)", "random_state": 42, "n_jobs": -1 }
  },
  {
    "model_name": "Dummy_MostFrequent",
    "library": "sklearn.dummy",
    "class": "DummyClassifier",
    "params": { "strategy": "most_frequent", "random_state": 42 }
  },
  {
    "model_name": "LabelSpreading_Default",
    "library": "sklearn.semi_supervised",
    "class": "LabelSpreading",
    "params": { "kernel": "rbf", "n_jobs": -1 }
  },
  {
    "model_name": "CalibratedClassifier_LinearSVC",
    "library": "sklearn.calibration",
    "class": "CalibratedClassifierCV",
    "params": { "estimator": "LinearSVC", "method": "sigmoid", "cv": 3 }
  },
  {
    "model_name": "CalibratedClassifier_Isotonic",
    "library": "sklearn.calibration",
    "class": "CalibratedClassifierCV",
    "params": { "estimator": "GaussianNB", "method": "isotonic", "cv": 3 }
  },
  {
    "model_name": "MLP_Layer100_50_Tanh",
    "library": "sklearn.neural_network",
    "class": "MLPClassifier",
    "params": { "hidden_layer_sizes": (100, 50), "activation": "tanh", "solver": "adam", "alpha": 0.001, "learning_rate_init": 0.001, "max_iter": 500, "random_state": 42 }
  },
  {
    "model_name": "MLP_Deep_Regularized",
    "library": "sklearn.neural_network",
    "class": "MLPClassifier",
    "params": { "hidden_layer_sizes": (100, 100, 100), "activation": "relu", "solver": "adam", "alpha": 0.01, "max_iter": 500, "random_state": 42 }
  },
  {
    "model_name": "MLP_Wide_DropoutLike",
    "library": "sklearn.neural_network",
    "class": "MLPClassifier",
    "params": { "hidden_layer_sizes": (200, 100), "activation": "relu", "solver": "adam", "alpha": 0.05, "max_iter": 500, "random_state": 42 }
  },
  {
    "model_name": "SVC_RBF_Default",
    "library": "sklearn.svm",
    "class": "SVC",
    "params": { "C": 1.0, "kernel": "rbf", "gamma": "scale", "probability": True, "random_state": 42 }
  },
  {
    "model_name": "SVC_RBF_Tight",
    "library": "sklearn.svm",
    "class": "SVC",
    "params": { "C": 10.0, "kernel": "rbf", "gamma": "scale", "probability": True, "random_state": 42 }
  },
  {
    "model_name": "SVC_Poly_Degree3",
    "library": "sklearn.svm",
    "class": "SVC",
    "params": { "C": 1.0, "kernel": "poly", "degree": 3, "probability": True, "random_state": 42 }
  },
  {
    "model_name": "NuSVC_Default",
    "library": "sklearn.svm",
    "class": "NuSVC",
    "params": { "nu": 0.5, "kernel": "rbf", "gamma": "scale", "probability": True, "random_state": 42 }
  },
  {
    "model_name": "LogisticRegression_L2_Default",
    "library": "sklearn.linear_model",
    "class": "LogisticRegression",
    "params": { "penalty": "l2", "C": 1.0, "solver": "lbfgs", "max_iter": 1000, "n_jobs": -1, "random_state": 42 }
  },
  {
    "model_name": "LogisticRegression_L1_Liblinear",
    "library": "sklearn.linear_model",
    "class": "LogisticRegression",
    "params": { "penalty": "l1", "C": 1.0, "solver": "liblinear", "max_iter": 1000, "random_state": 42 }
  },
  {
    "model_name": "LogisticRegression_ElasticNet",
    "library": "sklearn.linear_model",
    "class": "LogisticRegression",
    "params": { "penalty": "elasticnet", "C": 0.5, "l1_ratio": 0.5, "solver": "saga", "max_iter": 2000, "n_jobs": -1, "random_state": 42 }
  },
  {
    "model_name": "LogisticRegression_StrongReg",
    "library": "sklearn.linear_model",
    "class": "LogisticRegression",
    "params": { "penalty": "l2", "C": 0.01, "solver": "lbfgs", "max_iter": 1000, "n_jobs": -1, "random_state": 42 }
  },
  {
    "model_name": "RidgeClassifier_Default",
    "library": "sklearn.linear_model",
    "class": "RidgeClassifier",
    "params": { "alpha": 1.0, "random_state": 42 }
  },
  {
    "model_name": "RidgeClassifier_HighAlpha",
    "library": "sklearn.linear_model",
    "class": "RidgeClassifier",
    "params": { "alpha": 10.0, "random_state": 42 }
  },
  {
    "model_name": "SGD_Hinge_LinearSVM",
    "library": "sklearn.linear_model",
    "class": "SGDClassifier",
    "params": { "loss": "hinge", "penalty": "l2", "alpha": 0.0001, "max_iter": 1000, "n_jobs": -1, "random_state": 42 }
  },
  {
    "model_name": "SGD_Log_Logistic",
    "library": "sklearn.linear_model",
    "class": "SGDClassifier",
    "params": { "loss": "log_loss", "penalty": "elasticnet", "alpha": 0.001, "l1_ratio": 0.15, "max_iter": 1000, "n_jobs": -1, "random_state": 42 }
  },
  {
    "model_name": "LinearSVC_Default",
    "library": "sklearn.svm",
    "class": "LinearSVC",
    "params": { "penalty": "l2", "loss": "squared_hinge", "C": 1.0, "dual": False, "max_iter": 2000, "random_state": 42 }
  },
  {
    "model_name": "LinearSVC_L1_Primal",
    "library": "sklearn.svm",
    "class": "LinearSVC",
    "params": { "penalty": "l1", "loss": "squared_hinge", "C": 0.5, "dual": False, "max_iter": 2000, "random_state": 42 }
  },
  {
    "model_name": "KNN_5_Uniform",
    "library": "sklearn.neighbors",
    "class": "KNeighborsClassifier",
    "params": { "n_neighbors": 5, "weights": "uniform", "metric": "minkowski", "n_jobs": -1 }
  }
]

# Zapisanie do pliku
output_file = 'binary_class_models.json'
import numpy as np


# Usunięcie duplikatów na podstawie 'model_name'
seen = set()
unique_models = []
for m in models_portfolio:
    name = m.get("model_name")
    if name not in seen:
        seen.add(name)
        unique_models.append(m)
models_portfolio = unique_models

with open(output_file, 'w') as f:
    json.dump(models_portfolio, f, indent=2)

print(f"Pomyślnie wygenerowano plik: {output_file} z {len(models_portfolio)} modelami.")