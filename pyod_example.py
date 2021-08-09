from pyod.models.copod import COPOD


def ano_detect(train):
    clf = COPOD()
    clf.fit(train)


if __name__ == '__main__':
    ano_detect()
