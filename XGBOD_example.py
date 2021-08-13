from pyod.models.xgbod import XGBOD
from pyod.utils.data import evaluate_print, generate_data
from pyod.utils.example import visualize
from sklearn.model_selection import train_test_split


def xgbod_example(random_state=57):
    X, y = generate_data(train_only=True)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=random_state
    )

    # 一言でいうと
    # 外れ値予測のためのアンサンブル学習＋後段にXGBoost (Semi-superbised learning)

    # Extreme Gradient Boosting Outlier Detection (XGBOD)は以下の3段階で処理をする
    # 1. 新しいデータ表現を生成。データに複数の教師なし外れ値検出アルゴリズムを適用し、外れ値スコアを算出。
    # 2. 新しく算出した外れ値スコアの中から有用なスコアを選択。算出したスコアをデータに追加し、新しい特徴空間を生成。
    # 	スコアの選択方法は3つ
    # 		ランダムに選択
    # 		Accurate選択：ROCが高くなるものを選択
    # 		バランス選択：相関高いペアを抜いてスコアのサブセットをつくる
    # 	高次元にはバランス選択、特徴が少ないデータはAccurate選択が良い(論文より)
    # 3. 新しく生成した特徴空間でXGBoostを学習。その出力によって外れ値を予測。
    # 	XGBoostで算出される特徴量重要度を用いて、予測に使う外れ値スコアの選定をする場合も

    # 論文
    # https://arxiv.org/pdf/1912.00290.pdf

    clf_name = 'XGBOD'
    clf = XGBOD(random_state=random_state)
    clf.fit(x_train, y_train)

    y_train_pred = clf.labels_  # ラベル(0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # 外れ値スコア

    y_test_pred = clf.predict(x_test)  # 外れ値ラベル(0 or 1)
    y_test_scores = clf.decision_function(x_test)  # 外れ値スコア

    print('On Training Data:')
    evaluate_print(clf_name, y_train_pred, y_train_scores)
    print('On Test Data:')
    evaluate_print(clf_name, y_test_pred, y_test_scores)

    visualize(
        clf_name,
        x_train, y_train,
        x_test, y_test,
        y_train_pred, y_test_pred,
        show_figure=True, save_figure=False
    )


if __name__ == '__main__':
    xgbod_example()
