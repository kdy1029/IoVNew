from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import (
    LogisticRegression,
    PassiveAggressiveClassifier,
    Perceptron,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    Bidirectional,
    Concatenate,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    Input,
    LSTM,
)


def build_mlp(input_dim):
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(128, activation="relu"),
            Dropout(0.2),
            Dense(64, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_lstm(input_shape):
    model = Sequential(
        [
            Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
            Dropout(0.2),
            LSTM(32),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_lstm_with_id(input_shape, num_ids):
    seq_in = Input(shape=input_shape, name="seq_in")
    id_in = Input(shape=(), dtype="int32", name="id_in")

    x = Bidirectional(LSTM(64, return_sequences=True))(seq_in)
    x = Dropout(0.2)(x)
    x = LSTM(32)(x)

    id_emb = Embedding(input_dim=num_ids, output_dim=16, name="id_emb")(id_in)
    id_emb = Flatten()(id_emb)

    h = Concatenate()([x, id_emb])
    h = Dense(64, activation="relu")(h)
    out = Dense(1, activation="sigmoid")(h)

    model = Model(inputs=[seq_in, id_in], outputs=out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def classical_models(seed=42):
    return [
        ("LogReg(balanced)", LogisticRegression(solver="liblinear", class_weight="balanced", random_state=seed)),
        (
            "LogReg-L1(saga, balanced)",
            LogisticRegression(
                solver="saga",
                penalty="l1",
                C=1.0,
                class_weight="balanced",
                random_state=seed,
                max_iter=2000,
                n_jobs=-1,
            ),
        ),
        ("LinearSVC(balanced)", LinearSVC(class_weight="balanced", max_iter=5000, random_state=seed)),
        (
            "SGD-hinge(balanced)",
            SGDClassifier(loss="hinge", class_weight="balanced", max_iter=2000, tol=1e-3, random_state=seed),
        ),
        ("RidgeClassifier(balanced)", RidgeClassifier(class_weight="balanced", random_state=seed)),
        (
            "Perceptron(balanced)",
            Perceptron(penalty="l2", class_weight="balanced", random_state=seed, max_iter=2000, tol=1e-3),
        ),
        (
            "PassiveAggressive(balanced)",
            PassiveAggressiveClassifier(
                loss="hinge", C=1.0, class_weight="balanced", random_state=seed, max_iter=2000, tol=1e-3
            ),
        ),
        (
            "DecisionTree(shallow)",
            DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, class_weight="balanced", random_state=seed),
        ),
        (
            "ExtraTrees(shallow)",
            ExtraTreesClassifier(
                n_estimators=64,
                max_depth=8,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            ),
        ),
        ("GaussianNB", GaussianNB()),
    ]


def artifact_models(seed=42):
    return {
        "logreg": LogisticRegression(solver="liblinear", class_weight="balanced", random_state=seed),
        "gnb": GaussianNB(),
        "extratrees": ExtraTreesClassifier(
            n_estimators=100,
            max_depth=8,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    }


def dummy_baseline(seed=42):
    return DummyClassifier(strategy="most_frequent", random_state=seed)


def complement_nb():
    return ComplementNB()


def bernoulli_nb():
    return BernoulliNB()

