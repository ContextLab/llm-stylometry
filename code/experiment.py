from data_utils import sample_book_path
from constants import AUTHORS, CLEANED_DATA_DIR


class Experiment:
    def __init__(
        self,
        train_author,
        seed,
        tokenizer_name,
        n_train_tokens=643041,
        excluded_train_path=None,
        n_positions=1024,
        n_embd=128,
        n_layer=8,
        n_head=8,
        batch_size=16,
        lr=5e-5,
        stop_criteria={
            "train_loss": 3.0,
            "min_epochs": 500,
            "max_epochs": 10000,
        },
        resume_training=False,
    ):
        self.name = f"{train_author}_tokenizer={tokenizer_name}_seed={seed}"
        self.eval_paths = {author: sample_book_path(author, seed) for author in AUTHORS}
        self.excluded_train_path = self.eval_paths[train_author]
        if train_author in ["baum", "thompson"]:
            self.eval_paths.update(
                {
                    "non_oz_baum": CLEANED_DATA_DIR / "non_oz_baum" / "48778.txt",
                    "non_oz_thompson": CLEANED_DATA_DIR
                    / "non_oz_thompson"
                    / "the_princess_of_cozytown.txt",
                    "contested": CLEANED_DATA_DIR / "contested" / "30537.txt",
                }
            )

        self.train_author = train_author
        self.seed = seed
        self.tokenizer_name = tokenizer_name
        self.n_train_tokens = n_train_tokens
        self.excluded_train_path = excluded_train_path
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.batch_size = batch_size
        self.lr = lr
        self.stop_criteria = stop_criteria
        self.resume_training = resume_training
