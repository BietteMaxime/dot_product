import pandas
import sys

if __name__ == "__main__":
    data = pandas.DataFrame([float(i) for i in sys.stdin.readlines()])
    print(data.describe())
