import pandas as pd
import os


def update_loss_log(
    log_file_path, epochs_completed, train_author, loss_dataset, loss_value, seed
):
    """
    Update the loss logs with a single dataset loss value.
    """
    if os.path.exists(log_file_path):
        loss_logs_df = pd.read_csv(log_file_path)

        new_row = {
            "seed": seed,
            "train_author": train_author,
            "epochs_completed": int(epochs_completed),
            "loss_dataset": loss_dataset,
            "loss_value": loss_value,
        }

        loss_logs_df = pd.concat(
            [loss_logs_df, pd.DataFrame([new_row])], ignore_index=True
        )
    else:
        assert epochs_completed == 0, "Epochs completed must be 0 for a new log file."

        loss_logs_df = pd.DataFrame(
            [
                {
                    "seed": seed,
                    "train_author": train_author,
                    "epochs_completed": int(epochs_completed),
                    "loss_dataset": loss_dataset,
                    "loss_value": loss_value,
                }
            ]
        )

    loss_logs_df.to_csv(log_file_path, index=False)
