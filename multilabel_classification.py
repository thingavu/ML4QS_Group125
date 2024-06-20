import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb
import lightgbm as lgb
import optuna

def load_data(file_path):
    data = pd.read_csv(file_path)
    data = data.dropna()
    le_tone = LabelEncoder()
    data['tone'] = le_tone.fit_transform(data['tone'])
    data['language_binary'] = data['language'].apply(lambda x: 0 if x == 'en' else 1)
    return data, le_tone

def split_data(data):
    subject3_data = data[data['participant'] == 'subject3']
    subject3_train_sample = subject3_data.sample(frac=0.1, random_state=42)
    subject3_train_sample['participant'] = 'subject2'

    subject3_test_data = subject3_data.drop(subject3_train_sample.index)

    test_data_ru_tr = data[(data['language'].isin(['ru', 'tr']))]

    test_data_ru_tr_sampled = test_data_ru_tr.sample(frac=0.2, random_state=42)

    test_data = pd.concat([subject3_test_data, test_data_ru_tr_sampled])

    train_data = data.drop(test_data.index)
    train_data = pd.concat([train_data, subject3_train_sample])

    X_train = train_data.drop(columns=['language', 'tone', 'participant', 'script', 'language_binary'])
    y_train = train_data[['language_binary', 'tone']]
    X_test = test_data.drop(columns=['language', 'tone', 'participant', 'script', 'language_binary'])
    y_test = test_data[['language_binary', 'tone']]
    return X_train, X_test, y_train, y_test

def objective(trial, X_train, y_train, X_test, y_test):
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'booster': 'gbtree',
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
    }
    multi_target_xgb = MultiOutputClassifier(xgb.XGBClassifier(**param), n_jobs=-1)
    multi_target_xgb.fit(X_train, y_train)
    y_pred = multi_target_xgb.predict(X_test)
    
    accuracy_language = accuracy_score(y_test['language_binary'], y_pred[:, 0])
    accuracy_tone = accuracy_score(y_test['tone'], y_pred[:, 1])
    
    return (accuracy_language + accuracy_tone) / 2

def objective_lightgbm(trial, X_train, y_train, X_test, y_test):
    param = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.5, 0.9),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.5, 0.9),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100)
    }
    multi_target_lgb = MultiOutputClassifier(lgb.LGBMClassifier(**param), n_jobs=-1)
    multi_target_lgb.fit(X_train, y_train)
    y_pred = multi_target_lgb.predict(X_test)
    
    accuracy_language = accuracy_score(y_test['language_binary'], y_pred[:, 0])
    accuracy_tone = accuracy_score(y_test['tone'], y_pred[:, 1])
    
    return (accuracy_language + accuracy_tone) / 2

def run_xgboost(default_params, X_train, X_test, y_train):
    multi_target_xgb = MultiOutputClassifier(xgb.XGBClassifier(**default_params), n_jobs=-1)
    multi_target_xgb.fit(X_train, y_train)
    y_pred = multi_target_xgb.predict(X_test)    
    return y_pred, multi_target_xgb

def run_lightgbm(default_params, X_train, X_test, y_train):
    multi_target_lgb = MultiOutputClassifier(lgb.LGBMClassifier(**default_params), n_jobs=-1)
    multi_target_lgb.fit(X_train, y_train)
    y_pred = multi_target_lgb.predict(X_test)
    return y_pred, multi_target_lgb

def plot_feature_importance(model, feature_names, model_name, filename):
    importances = None
    
    if isinstance(model.estimator, xgb.XGBClassifier):
        importances = [est.feature_importances_ for est in model.estimators_]
    elif isinstance(model.estimator, lgb.LGBMClassifier):
        importances = [est.feature_importances_ for est in model.estimators_]

    if importances is not None:
        avg_importances = sum(importances) / len(importances)
        feature_importances = pd.DataFrame({
            'feature': feature_names,
            'importance': avg_importances
        })
        feature_importances = feature_importances.sort_values(by='importance', ascending=False).head(10)

        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importances, palette='viridis')
        plt.xlabel('Feature Importance', fontsize=18)
        plt.ylabel('Feature', fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)        
        plt.title(f'Top 10 Feature Importances for {model_name}', fontsize=18)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        for index, value in enumerate(feature_importances['importance']):
            plt.text(value, index, f'{value:.2f}', color='black', ha="right", va="center", fontsize=12)

        plt.savefig(filename, bbox_inches='tight')
        plt.close()

def save_confusion_matrix(cm, labels, title, filename):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='viridis')
    plt.title(title)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def main(model, path, fine_tune=True):
    data, le_tone = load_data(path)

    X_train, X_test, y_train, y_test = split_data(data)    

    default_xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'booster': 'gbtree',
        'learning_rate': 0.2118,
        'max_depth': 6,
        'min_child_weight': 1,
        'gamma': 9.143e-06,
        'subsample': 0.9578,
        'colsample_bytree': 0.8402,
    }
    
    default_lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.1,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20
    }

    if fine_tune:
        study = optuna.create_study(direction='maximize')
        if model == 'xgboost':
            study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=50)
            best_params = study.best_params
            print("Best XGBoost Parameters:", best_params)
            y_pred, best_model = run_xgboost(best_params, X_train, X_test, y_train)
        else:
            study.optimize(lambda trial: objective_lightgbm(trial, X_train, y_train, X_test, y_test), n_trials=50)
            best_params = study.best_params
            print("Best LightGBM Parameters:", best_params)
            y_pred, best_model = run_lightgbm(best_params, X_train, X_test, y_train)
    else:
        if model == 'xgboost':
            y_pred, best_model = run_xgboost(default_xgb_params, X_train, X_test, y_train)
        else:
            y_pred, best_model = run_lightgbm(default_lgb_params, X_train, X_test, y_train)

    report_language = classification_report(y_test['language_binary'], y_pred[:, 0], target_names=['non-native', 'native'])
    report_tone = classification_report(y_test['tone'], y_pred[:, 1], target_names=le_tone.classes_)

    print("Classification Report for Language (Native vs Non-native):")
    print(report_language)

    print("Classification Report for Tone:")
    print(report_tone)

    plot_feature_importance(best_model, X_train.columns, model, f'{model}_feature_importance.png')

    # Plot and save confusion matrices
    cm_language = confusion_matrix(y_test['language_binary'], y_pred[:, 0])
    cm_tone = confusion_matrix(y_test['tone'], y_pred[:, 1])

    save_confusion_matrix(cm_language, ['non-native', 'native'], 'Confusion Matrix for Language (Native vs Non-native)', 'confusion_matrix_language.png')
    save_confusion_matrix(cm_tone, le_tone.classes_, 'Confusion Matrix for Tone', 'confusion_matrix_tone.png')

if __name__ == '__main__':
    main('xgboost', './data_w_features_ver2/data_w_all_features_final.csv', fine_tune=False)