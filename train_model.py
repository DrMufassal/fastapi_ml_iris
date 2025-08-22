import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from datetime import datetime

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

def main():
    iris = load_iris()
    X = iris.data
    y = iris.target
    class_names = iris.target_names.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    metadata = {
        "model_type": "LogisticRegression in sklearn Pipeline with StandardScaler",
        "problem_type": "classification",
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "features": FEATURES,
        "classes": class_names,
        "metrics": {"test_accuracy": round(float(acc), 4)}
    }

    artifact = {"pipeline": pipe, "metadata": metadata}
    joblib.dump(artifact, "model.pkl")
    print("Saved model.pkl")
    print("Test accuracy:", acc)

if __name__ == "__main__":
    main()