from flask import Flask, render_template, request,jsonify
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
app = Flask(__name__)

def sort_pairs_in_row(row):
    # Chuyển đổi mỗi dòng thành các cặp số
    pairs = [[row[i], row[i+1]] for i in range(0, len(row), 2)]
    # Sắp xếp các cặp số trong mỗi dòng
    sorted_pairs = sorted(pairs, key=lambda x: (x[1], x[0]))
    return sorted_pairs

def sort_data(data):
    data = pd.DataFrame(data)
    sorted_data = data.apply(sort_pairs_in_row, axis=1)
    sorted_data = np.array(sorted_data.to_list()).reshape(-1, 10)
    return sorted_data

def train_model(): 
    data = pd.read_csv("./data/poker-hand-training-true.data", delimiter=",", header=None)
    print("Số dòng tập dữ liệu: ",len(data))
    
    kf = KFold(n_splits=10)
    X= sort_data(data.iloc[:,0:10])
    Y = data.iloc[:, -1]
    
    modalResults = None
    f1Avg = 0
    f1Max = 0
    for train_index, test_index in kf.split(data):
        print("Train:", train_index, "Test:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        array_neighbor = [3, 5, 7, 9, 11, 13, 15]
        #Tìm n_neighbor có f1 lớn nhất
        n_neighborMax = 0
        f1_score_KNNMax = 0
        for n_neighbor in array_neighbor:
            knn = KNeighborsClassifier(n_neighbors=n_neighbor)
            knn.fit(X_train, Y_train)
            Y_pred_KNN = knn.predict(X_test)
            f1_score_KNN = round(f1_score(Y_test, Y_pred_KNN, average='macro', zero_division=1),4)
            
            if(f1_score_KNNMax < f1_score_KNN):
                f1_score_KNNMax = f1_score_KNN
                n_neighborMax = n_neighbor
        print("Chỉ số F1 KNN =", f1_score_KNNMax, " ở n neighbor =", n_neighborMax)
        bayes = GaussianNB()
        bayes.fit(X_train, Y_train)
        clf = DecisionTreeClassifier(criterion='gini', random_state=42)
        clf.fit(X_train, Y_train)
        Y_pred_Bayes = bayes.predict(X_test)
        Y_pred_Clf = clf.predict(X_test)
        f1_score_Bayes = round(f1_score(Y_test, Y_pred_Bayes, average='macro', zero_division=1),4)
        f1_score_Clf = round(f1_score(Y_test, Y_pred_Clf, average='macro', zero_division=1),4)
        f1Avg += f1_score_Clf
        print(classification_report(Y_test,Y_pred_Clf,   zero_division=1))
        print("Chỉ số F1 Bayes =",  f1_score_Bayes)
        print("Chỉ số F1 Tree =",  f1_score_Clf," độ sâu của cây", clf.tree_.max_depth)
        if(f1Max < f1_score_Clf):
            modalResults = clf
    print("Chỉ số F1 trung bình =", round(f1Avg/10 *100, 4) )
    return modalResults

@app.route("/")
def main():
    return render_template('index.html')

@app.route('/forecast/', methods=['POST'])
def get_post_data(): 
    data = request.get_json()
    data_test = sort_data([data])
    print(data_test)
    pred = clf.predict(data_test).tolist()
    message = ""
    match pred[0]: 
        case 0: message = "Bài của bạn không có gì cả, hãy diễn đi nào!" 
        case 1: message = "Bài của bạn có một đôi"
        case 2: message = "Bài của bạn có hai đôi"
        case 3: message = "Bài của bạn có xám cô"
        case 4: message = "Bài của bạn có sảnh"
        case 5: message = "Bài của bạn có thùng"
        case 6: message = "Bài của bạn có cù lũ"
        case 7: message = "Wow, Bài của bạn có tứ quý chơi tất nào"
        case 8: message = "Wow, Bài của bạn là thùng phá sảnh, chơi tất nào"
        case 9: message = "Bài của bạn là thùng phá sảnh rồng, ALL IN"
    return jsonify(message)


clf= train_model()
app.run()