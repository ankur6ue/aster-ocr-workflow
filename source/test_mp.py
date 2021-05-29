from multiprocessing import Pool
def doubler(number, _):
    return number * 2
if __name__ == '__main__':
    with Pool(processes=3) as pool:
        result = pool.starmap(doubler, [(25, 1), (30, 1), (60, 1), (80, 1)])
        print(result)