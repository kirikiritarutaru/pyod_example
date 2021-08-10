import warnings

from pyod.models.xgbod import XGBOD
from pyod.utils.data import evaluate_print, generate_data
from sklearn.model_selection import train_test_split

warnings.simplefilter('ignore')


def xgbod_example(random_state=57):
    X, y = generate_data(train_only=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=random_state
    )

    clf_name = 'XGBOD'
    clf = XGBOD(random_state=random_state)
    clf.fit(X_train, y_train)

    y_train_pred = clf.labels_  # ラベル(0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # 外れ値スコア

    y_test_pred = clf.predict(X_test)  # 外れ値ラベル(0 or 1)
    y_test_scores = clf.decision_function(X_test)  # 外れ値スコア

    print('On Training Data:')
    evaluate_print(clf_name, y_train, y_train_scores)
    print('On Test Data:')
    evaluate_print(clf_name, y_test, y_test_scores)


if __name__ == '__main__':
    xgbod_example()
