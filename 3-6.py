from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.cross_validation import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import myplot as plt2

def main():
    #datasets
    iris = datasets.load_iris()
    X, Y = iris.data[:, [2,3]], iris.target
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    #tree
    tree = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=0)
    tree.fit(x_train, y_train)
    
    plt2.plot_decision_regions(X, Y, classifier=tree, test_idx=(105, 150))
    plt.xlabel("gaku")
    plt.ylabel("length")
    plt.legend(loc="upper left")
    plt.show()

    export_graphviz(tree, out_file="tree.dot", feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True)
    # dot -Teps tree.dot -o tree.png

if __name__ == "__main__":
    main()
