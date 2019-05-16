import matplotlib.pyplot as plt


if __name__ == '__main__':
    x = []
    y = []
    z = []
    with open('data.txt', 'r') as fin:
        for line in fin:
            for v, t, l in zip(line.split(), (int, int, float), (x, y, z)):
                l.append(t(v))

    lx, rx = min(x), max(x)
    ly, ry = min(y), max(y)

    arr = [
        [0] * (ry - ly + 1)
        for _ in range(rx - lx + 1)
    ]
    for i in range(len(x)):
        arr[x[i] - lx][y[i] - ly] = z[i]

    plt.imshow(arr, cmap='hot', interpolation='nearest')
    plt.show()