import numpy as np
from sklearn.datasets import load_digits
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from pandas import DataFrame

X, y = load_digits(return_X_y=True)

ss = StandardScaler()
pca = PCA(n_components=0.8)
skb = SelectKBest(k=int(np.sqrt(X.shape[1])))

clfs = [GaussianNB(), KNeighborsClassifier(), DecisionTreeClassifier()]

rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5)
accuracy_scores= np.zeros((5*2, len(clfs)))
accuracy_scores_ss= np.zeros((5*2, len(clfs)))
accuracy_scores_pca= np.zeros((5*2, len(clfs)))
accuracy_scores_skb= np.zeros((5*2, len(clfs)))


for fold, (train_index, test_index) in enumerate(rskf.split(X, y)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    ss.fit(X_train, y_train)
    X_train_ss = ss.transform(X_train)
    X_test_ss = ss.transform(X_test)

    pca.fit(X_train, y_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    skb.fit(X_train, y_train)
    X_train_skb = skb.transform(X_train)
    X_test_skb = skb.transform(X_test)


    for clf_id, clf in enumerate(clfs):
        y_pred = clone(clf).fit(X_train, y_train).predict(X_test)
        accuracy_scores[fold ,clf_id] = accuracy_score(y_test, y_pred)

        y_pred_ss = clone(clf).fit(X_train_ss, y_train).predict(X_test_ss)
        accuracy_scores_ss[fold ,clf_id] = accuracy_score(y_test, y_pred_ss)

        y_pred_pca = clone(clf).fit(X_train_pca, y_train).predict(X_test_pca)
        accuracy_scores_pca[fold ,clf_id] = accuracy_score(y_test, y_pred_pca)

        y_pred_skb = clone(clf).fit(X_train_skb, y_train).predict(X_test_skb)
        accuracy_scores_skb[fold ,clf_id] = accuracy_score(y_test, y_pred_skb)
        
mean = np.mean(accuracy_scores, axis=0)
mean_ss = np.mean(accuracy_scores_ss, axis=0)
mean_pca = np.mean(accuracy_scores_pca, axis=0)
mean_skb = np.mean(accuracy_scores_skb, axis=0)
std = np.std(accuracy_scores, axis=0)
std_ss = np.std(accuracy_scores_ss, axis=0)
std_pca = np.std(accuracy_scores_pca, axis=0)
std_skb = np.std(accuracy_scores_skb, axis=0)

def print_results(title, mean, std):
    print(f"\n{title}")
    for clf_id, clf_name in enumerate(['GNB', 'KNN', 'DT']):
        print(clf_name)
        print("Mean: {:0.3f}".format(mean[clf_id]))
        print("Standard deviation: {:0.3f}".format(std[clf_id]))

print_results("Base dataset", mean, std)
print_results("Standard scaler", mean_ss, std_ss)
print_results("PCA", mean_pca, std_pca)
print_results("SelectKBest", mean_skb, std_skb)

print("\nMean scores:")
d = {'Base': [mean[0], mean[1], mean[2]], 'Norm': [mean_ss[0], mean_ss[1], mean_ss[2]], 'PCA': [mean_pca[0], mean_pca[1], mean_pca[2]], 'SKB': [mean_skb[0], mean_skb[1], mean_skb[2]]}
df = DataFrame(data=d, index=['GNB', 'KNN', 'DT'])
print(df)


