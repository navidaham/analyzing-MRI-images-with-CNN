import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.train_ann import run_ann_experiment  # type: ignore

if __name__ == "__main__":
    metrics = run_ann_experiment(
        name="final_ann",        # new name for final model
        hidden_layers=(64,),     # your best layout
        activation="leakyrelu",  # example – put your best here
        learning_rate=1e-4,      # best lr
        batch_size=32,
        max_epochs=80,
        early_stopping_patience=7,
        use_batchnorm=True,
        use_dropout=True,
        dropout_rate=0.5,
        augment=True,
        save_weights=True,
    )
    print("Final ANN training done. exp_id:", metrics["exp_id"])
