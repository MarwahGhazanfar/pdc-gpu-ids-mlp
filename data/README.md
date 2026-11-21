# Data

Dataset files are **not stored in the repository**.

1. Download the CICIDS2017 `MachineLearningCSV` files from https://www.unb.ca/cic/datasets/ids-2017.html
2. Place the CSV files in this folder.
3. Record SHA-256 checksums in `checksums.txt` (`sha256sum *.csv > checksums.txt`) and verify later with
   `sha256sum -c checksums.txt`.
4. Run the preprocessing pipeline in `preprocess/` to create the float32 binary arrays used by all backends.
