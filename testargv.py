import sys
print(sys.argv)
sys.argv.append("dsaf")
sys.argv.append("dsafa")
sys.argv.append("dsaf")
sys.argv.append("-d-saf")

print(sys.argv)

from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("simplescaling/s1K-1.1_tokenized")
