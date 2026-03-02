import pandas as pd
import glob
print(f"Pandas version: {pd.__version__}")
files = glob.glob("data/things-meg-ds004212/**/*events.tsv", recursive=True)
for f in files:
    try:
        # The engine='python' fallback might be triggering a utf-8 decode on the raw file handle
        # before 'latin-1' is applied to the string buffer
        df = pd.read_csv(f, sep='\t', encoding='latin-1', engine='c', on_bad_lines='skip')
    except Exception as e:
        print(f"Error on {f}: {e}")
        break
