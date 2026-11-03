from datasets import load_dataset

ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train[:50000]")  # first 6000 articles
print(ds)                      # columns: article, highlights, id
print(ds[0]["article"][:100])  # quick look

ds.to_csv("cnn_dailymail_sample.csv")   # for use with --input