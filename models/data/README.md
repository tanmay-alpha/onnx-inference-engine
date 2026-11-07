# Training data

No dataset is committed. The fraud model in `models/fraud/` was trained
on synthetic data generated inside `models/fraud/train_fraud_model.py`, so
reproducing it needs no download.

To train on real data, download the PaySim CSV from Kaggle
([`ealaxi/paysim1`](https://www.kaggle.com/datasets/ealaxi/paysim1)) into this
directory, for example with the Kaggle CLI:

```bash
kaggle datasets download -d ealaxi/paysim1 -p models/data --unzip
python models/fraud/train_fraud_model.py --paysim-csv models/data/PS_20174392719_1491204439457_log.csv --out-dir /tmp/fraud-paysim
```

The script reads the `type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`,
`oldbalanceDest`, `newbalanceDest` and `isFraud` columns. See
`models/fraud/README.md` for the feature encoding. Do not commit the CSV.
