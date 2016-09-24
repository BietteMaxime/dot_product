import pandas
import sys

pandas.options.display.width = 180

# DATA = """StaticVec 00 size 8192 nb_unitvecs 1024 9.15925e+10 elapsed 23661 ns
# StaticVec 01 9.15925e+10 elapsed 4717 ns
# StaticVec 02 9.15924e+10 elapsed 1988 ns
# StaticVec 03 9.15923e+10 elapsed 3660 ns
# StaticVec 00 9.15925e+10 elapsed 14357 ns
# StaticVec 01 9.15925e+10 elapsed 5198 ns
# StaticVec 02 9.15924e+10 elapsed 2681 ns
# StaticVec 03 9.15923e+10 elapsed 3335 ns""".splitlines()
DATA = sys.stdin.readlines()

if __name__ == "__main__":
    data = [i.split() for i in DATA]
    data = pandas.DataFrame([(int(i[1]), float(i[4])) for i in data], columns=["method", "ns"])
    ret = list()
    for method in data["method"].unique():
        desc = data[data.method == method].describe().transpose().ix[1]
        ret.append(pandas.Series(method, ["method",]).append(desc))
    print(pandas.DataFrame(ret))
