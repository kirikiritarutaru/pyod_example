from pyod.models.sod import SOD
from pyod.utils.data import evaluate_print, generate_data
from pyod.utils.example import visualize


def sod_example(
    n_train: int = 200,
    n_test: int = 100,
    contamination: float = 0.1
):
    x_train, x_test, y_train, y_test = generate_data(
        n_train=n_train,
        n_test=n_test,
        contamination=contamination,
        behaviour='new',
        random_state=57
    )

    # Subspace outlier detection (SOD)
    # アルゴリズムの特徴：入力は高次元のデータを想定

    # 現実のデータはえてして高次元。
    # 膨大な独立変数の中から関連ある特徴をサブグループにまとめて
    # originalの特徴空間の部分空間に射影することはよくあるよね
    # →アルゴリズムにしちゃおう

    # inlierの各データポイントは、ある軸に平行な超平面（←部分空間）上にのると仮定
    # 上記超平面から外れたオブジェクトは異常とみなす

    # やってることkNNでまとめてPCAみたいな

    # 参考：
    # https://nandhini-aitec.medium.com/day-28-subspace-outlier-detection-sod-2bdfd7e60343
    clf_name = 'SOD'
    clf = SOD()
    clf.fit(x_train)

    y_train_pred = clf.labels_
    y_train_scores = clf.decision_scores_

    y_test_pred = clf.predict(x_test)  # 外れ値ラベル(0 or 1)
    y_test_scores = clf.decision_function(x_test)  # 外れ値スコア

    print('On Training Data:')
    evaluate_print(clf_name, y_train, y_train_scores)
    print('On Test Data:')
    evaluate_print(clf_name, y_test, y_test_scores)
    visualize(
        clf_name,
        x_train, y_train,
        x_test, y_test,
        y_train_pred, y_test_pred,
        show_figure=True, save_figure=False
    )


if __name__ == '__main__':
    sod_example()
