import warnings

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

def run_optuna(X, y, n_trials=50):
    def objective(trial):
        # Suggest hyperparameters
        penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
        C = trial.suggest_float("C", 1e-5, 1e2, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0) if penalty == "elasticnet" else None
        solver = "saga" if penalty in ["l1", "elasticnet"] else "lbfgs"
        threshold = trial.suggest_float("selection_threshold", 1e-4, 0.5, log=True)

        model = LogisticRegression(
            penalty=penalty,
            C=C,
            solver=solver,
            l1_ratio=l1_ratio,
            max_iter=1000,
            random_state=42
        )
        selector = SelectFromModel(estimator=model, threshold=threshold)

        # Define pipeline
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("feature_selector", selector),
            ("clf", model)
        ])

        try:
            score = cross_val_score(
                pipeline,
                X,
                y,
                cv=5,
                scoring="roc_auc",
                n_jobs=-1,
                error_score="raise"
            )
            return np.mean(score)
        except ValueError as e:
            if "0 feature(s)" in str(e):
                raise optuna.exceptions.TrialPruned()
            else:
                raise

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study

def plot_roc_curve(y_true, y_scores, title="ROC Curve", save_path=None, filename=None):
   
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid()

    return fig


def build_lr_pipeline_from_best_params(best_params):
    """
    Build a pipeline using the best hyperparameters from Optuna.
    """
    solver = "saga" if best_params["penalty"] in ["l1", "elasticnet"] else "lbfgs"

    model = LogisticRegression(
        penalty=best_params["penalty"],
        C=best_params["C"],
        solver=solver,
        l1_ratio=best_params.get("l1_ratio"),
        max_iter=1000,
        random_state=42
    )

    selector = SelectFromModel(estimator=model, threshold=best_params["selection_threshold"])

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("feature_selector", selector),
        ("clf", model)
    ])

    return pipeline

def train_nn_and_crossval_predict(X, y, n_epochs) -> np.ndarray:
    """
    Train a neural network with dropout and weight decay, perform cross-validation prediction,
    and plot combined training/validation loss curves across all folds.
    Returns the out-of-fold predictions.
    """


    class SimpleNN(nn.Module):
        def __init__(self, input_dim):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_dim, 128)
            self.dropout1 = nn.Dropout(p=0.5)
            self.fc2 = nn.Linear(128, 64)
            self.dropout2 = nn.Dropout(p=0.5)
            self.fc3 = nn.Linear(64, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout1(x)
            x = torch.relu(self.fc2(x))
            x = self.dropout2(x)
            x = self.fc3(x)
            return x

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros((X.shape[0],))

    all_train_losses = []
    all_val_losses = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n🔁 Fold {fold}/5")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = SimpleNN(X.shape[1]).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        train_losses = []
        val_losses = []

        for epoch in range(n_epochs):
            # Training
            model.train()
            inputs = torch.tensor(X_train, dtype=torch.float32).to(device)
            targets = torch.tensor(y_train, dtype=torch.float32).to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            # Validation
            model.eval()
            with torch.no_grad():
                val_inputs = torch.tensor(X_val, dtype=torch.float32).to(device)
                val_targets = torch.tensor(y_val, dtype=torch.float32).to(device).unsqueeze(1)
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_targets)
                val_losses.append(val_loss.item())

        # Store loss history for plotting later
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

        # Final predictions for this fold
        model.eval()
        with torch.no_grad():
            val_outputs = torch.sigmoid(model(torch.tensor(X_val, dtype=torch.float32).to(device)))
            oof_preds[val_idx] = val_outputs.cpu().numpy().reshape(-1)

    # Plot combined loss curves
    plt.figure(figsize=(10, 6))
    for i in range(5):
        plt.plot(all_train_losses[i], label=f"Train Loss Fold {i+1}", linestyle='--')
        plt.plot(all_val_losses[i], label=f"Val Loss Fold {i+1}")
    plt.title("Training & Validation Loss Across Folds")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return oof_preds


