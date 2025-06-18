import datasets

myset = datasets.DatasetBuilder.as_dataset(
    path="my_dataset",
    split="train",
    streaming=True,
)

