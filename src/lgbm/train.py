import click
import joblib
import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.metrics import mean_squared_error


def use_only_score_dist(
	semantic_retrieval_df: pd.DataFrame,
	lexical_retrieval_df: pd.DataFrame,
	upper_bound_df: pd.DataFrame,
	k: int = 20,
):
	X = pd.DataFrame(
		{
			"query": upper_bound_df["query"].tolist(),
		}
	)
	for idx in range(k):
		X[f"{idx}_semantic_score"] = semantic_retrieval_df["retrieve_scores"].apply(
			lambda x: x[idx]
		)
		X[f"{idx}_lexical_score"] = lexical_retrieval_df["retrieve_scores"].apply(
			lambda x: x[idx]
		)

	y = upper_bound_df["best_weight"]

	X.drop(columns=["query"], inplace=True)

	# Train
	sliced_idx = int(len(X) * 0.85)
	X_train = X[:sliced_idx]
	X_val = X[sliced_idx:]
	y_train = y[:sliced_idx]
	y_val = y[sliced_idx:]

	model = LGBMRegressor(
		n_estimators=1000,
		learning_rate=0.005,
		max_depth=5,
		random_state=42,
	)

	eval_set = [(X_train, y_train), (X_val, y_val)]

	model.fit(X_train, y_train, eval_set=eval_set, eval_metric="mse")

	y_pred = model.predict(X_val)
	mse = mean_squared_error(y_val, y_pred)

	print(f"Validation MSE: {mse}")

	return model


def use_score_dist_classify(
	semantic_retrieval_df: pd.DataFrame,
	lexical_retrieval_df: pd.DataFrame,
	upper_bound_df: pd.DataFrame,
	k: int = 20,
):
	X = pd.DataFrame(
		{
			"query": upper_bound_df["query"].tolist(),
		}
	)
	for idx in range(k):
		X[f"{idx}_semantic_score"] = semantic_retrieval_df["retrieve_scores"].apply(
			lambda x: x[idx]
		)
		X[f"{idx}_lexical_score"] = lexical_retrieval_df["retrieve_scores"].apply(
			lambda x: x[idx]
		)

	y = (upper_bound_df["best_weight"] > 0.5).astype(int)

	X.drop(columns=["query"], inplace=True)

	# Train
	sliced_idx = int(len(X) * 0.85)
	X_train = X[:sliced_idx]
	X_val = X[sliced_idx:]
	y_train = y[:sliced_idx]
	y_val = y[sliced_idx:]

	model = LGBMClassifier(
		n_estimators=1000,
		learning_rate=0.005,
		max_depth=5,
		random_state=42,
	)

	model.fit(X_train, y_train)

	y_pred = model.predict(X_val)
	y_pred_proba = model.predict_proba(X_val)[:, 1]

	from sklearn.metrics import (
		accuracy_score,
		precision_score,
		recall_score,
		f1_score,
		roc_auc_score,
	)

	accuracy = accuracy_score(y_val, y_pred)
	precision = precision_score(y_val, y_pred)
	recall = recall_score(y_val, y_pred)
	f1 = f1_score(y_val, y_pred)
	auc = roc_auc_score(y_val, y_pred_proba)

	print("Validation Metrics:")
	print(f"Accuracy: {accuracy:.4f}")
	print(f"Precision: {precision:.4f}")
	print(f"Recall: {recall:.4f}")
	print(f"F1 Score: {f1:.4f}")
	print(f"AUC-ROC: {auc:.4f}")

	return model


@click.command()
@click.option(
	"--semantic_retrieval_df_path",
	type=str,
	required=True,
	help="Path to the semantic retrieval dataframe",
)
@click.option(
	"--lexical_retrieval_df_path",
	type=str,
	required=True,
	help="Path to the lexical retrieval dataframe",
)
@click.option(
	"--upper_bound_df_path",
	type=str,
	required=True,
	help="Path to the upper bound dataframe",
)
@click.option(
	"--k",
	type=int,
	default=20,
	help="Number of top scores to use as features",
)
@click.option(
	"--model_save_path",
	type=str,
	required=True,
	help="Path to save the trained model",
)
@click.option("--mode", type=click.Choice(["regression", "classification"]))
def cli(
	semantic_retrieval_df_path: str,
	lexical_retrieval_df_path: str,
	upper_bound_df_path: str,
	k: int,
	model_save_path: str,
	mode,
):
	semantic_retrieval_df = pd.read_parquet(
		semantic_retrieval_df_path, engine="pyarrow"
	)
	lexical_retrieval_df = pd.read_parquet(lexical_retrieval_df_path, engine="pyarrow")
	upper_bound_df = pd.read_parquet(upper_bound_df_path, engine="pyarrow")

	if mode == "regression":
		model = use_only_score_dist(
			semantic_retrieval_df, lexical_retrieval_df, upper_bound_df, k
		)
	elif mode == "classification":
		model = use_score_dist_classify(
			semantic_retrieval_df, lexical_retrieval_df, upper_bound_df, k
		)
	else:
		raise ValueError("Invalid mode")

	joblib.dump(model, model_save_path)
	print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
	cli()
