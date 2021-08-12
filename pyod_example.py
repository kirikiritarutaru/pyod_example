import numpy as np
from pyod.models.combination import aom, average, maximization, median, moa
from pyod.models.knn import KNN
from pyod.utils.data import evaluate_print, generate_data
from pyod.utils.example import visualize
from pyod.utils.utility import standardizer
from sklearn.model_selection import train_test_split


def knn_example(
    n_train: int = 200,  # 学習データのデータ数
    n_test: int = 100,  # テストデータのデータ数
    contamination: float = 0.1  # 外れ値の割合（％）
):
    X_train, X_test, y_train, y_test = generate_data(
        n_train=n_train, n_test=n_test,
        contamination=contamination,
        random_state=57, behaviour='new'
    )

    clf_name = 'KNN'
    clf = KNN()
    clf.fit(X_train)

    # 学習データの予測ラベルと外れ値のスコアを取得
    y_train_pred = clf.labels_  # 予測ラベル(0: inlier, 1: outlier)
    y_train_scores = clf.decision_scores_  # 外れ値のスコア

    # テストデータの予測を取得
    y_test_pred = clf.predict(X_test)  # 外れ値のラベル(0 or 1)
    y_test_scores = clf.decision_function(X_test)  # 外れ値のスコア

    # 結果の評価
    print('On Training Data:')
    evaluate_print(clf_name, y_train, y_train_scores)
    print('On Test Data:')
    evaluate_print(clf_name, y_test, y_test_scores)

    visualize(
        clf_name,
        X_train, y_train,
        X_test, y_test,
        y_train_pred, y_test_pred,
        show_figure=True, save_figure=False
    )


def ensemble_example(n_clf: int = 20):
    # 外れ値検出は、教師なし学習のためモデルが不安定になることが多い
    # ロバスト性を改善には、検出器の出力を組み合わせる（たとえば、平均化）ことが考えられる
    X, y = generate_data(train_only=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    X_train_norm, X_test_norm = standardizer(X_train, X_test)

    k_list = list(range(10, 200+1, 10))

    train_scores = np.zeros([X_train.shape[0], n_clf])
    test_scores = np.zeros([X_test.shape[0], n_clf])

    print(f'Combining {n_clf} kNN detectors')
    for i in range(n_clf):
        clf = KNN(n_neighbors=k_list[i], method='largest')
        clf.fit(X_train_norm)

        train_scores[:, i] = clf.decision_scores_
        test_scores[:, i] = clf.decision_function(X_test_norm)

    train_scores_norm, test_scores_norm = standardizer(
        train_scores, test_scores
    )

    # 全モデルの平均スコアによるアンサンブル
    y_by_average = average(test_scores_norm)
    evaluate_print('Combination by Average', y_test, y_by_average)

    # 全モデルの最大スコアによるアンサンブル
    y_by_maximazation = maximization(test_scores_norm)
    evaluate_print('Combination by Maximization', y_test, y_by_maximazation)

    # 全モデルの中央値スコアによるアンサンブル
    y_by_median = median(test_scores_norm)
    evaluate_print('Combination by Median', y_test, y_by_median)

    # 全モデルをサブグループにわけたときのスコアの最大値の平均によるアンサンブル
    # 1. モデルをサブグループに分割
    # 2. 各サブグループのスコアの最大値をとる
    # 3. 2の手順でとったスコアの平均をとる
    y_by_aom = aom(test_scores_norm, n_buckets=5)
    evaluate_print('Combination by AOM', y_test, y_by_aom)

    # 全モデルをサブグループにわけたときのスコアの最大値の平均によるアンサンブル
    # 1. モデルをサブグループに分割
    # 2. 各サブグループのスコアの平均値をとる
    # 3. 2の手順でとったスコアの最大値をとる
    y_by_moa = moa(test_scores_norm, n_buckets=5)
    evaluate_print('Combination by MOA', y_test, y_by_moa)


if __name__ == '__main__':
    # knn_example()
    ensemble_example()
