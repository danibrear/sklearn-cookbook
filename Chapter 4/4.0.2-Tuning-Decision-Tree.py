from StringIO import StringIO
from sklearn import tree
import pydot
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

X, y = datasets.make_classification(1000, 20, n_informative=3)
dt = DecisionTreeClassifier()
dt.fit(X, y)

str_buffer = StringIO()
tree.export_graphviz(dt, out_file=str_buffer)
graph = pydot.graph_from_dot_data(str_buffer.getvalue())
graph.write('myfile.jpg')
