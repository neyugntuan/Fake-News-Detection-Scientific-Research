from src.dataset import load_data
from src.experiment_dual import run_dual_experiment


def main():

    # Đường dẫn: dataset_fullEDA.csv (output của preprocessing_eda_pipeline.py)
    X, y, vocab_size, vocab_dict, lengths = load_data("data/dataset_fullEDA.csv")

    print(f"\n📊 Dataset: {len(X)} samples | Vocab: {vocab_size} | Max len: {X.shape[1]}")

    run_dual_experiment(X, y, vocab_size, vocab_dict)


if __name__ == "__main__":
    main()
