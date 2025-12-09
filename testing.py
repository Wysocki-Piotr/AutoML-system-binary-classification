from sklearn.base import ClassifierMixin
from sklearn.utils import all_estimators
classifiers=[est for est in all_estimators() if issubclass(est[1], ClassifierMixin)]

# Print all classifier algorithms to file testing.txt
with open("testing.txt", "w") as f:
    for clas in classifiers:
        print(clas[0])
        f.write(f"{clas[0]}\n")
