import numpy as np

def Gauss_Elimination(matrix, result):
    # Forward Elimination
    for k in range(len(matrix) - 1):
        for i in range(k+1, len(matrix)):
            factor = matrix[i][k] / matrix[k][k]
            for j in range(len(matrix)):
                matrix[i][j] = matrix[i][j] - factor * matrix[k][j]
            result[i] = result[i] - factor * result[k]
    # backward substitution
    list_x = [0, 0, 0, 0]
    for i in reversed(range(len(matrix))):
        sum = 0
        for j in range(i, len(matrix)):
            if i != len(matrix) - 1:
                sum = sum + matrix[i][j] * list_x[j]
            list_x[i] = (result[i] - sum) / matrix[i][i]
    return list_x

def LU_Decomposition(matrix, result):
    # Obtain L and U matrix
    n = matrix.shape[0]
    u = np.copy(matrix)
    l = np.eye(n)
    for i in range(n):
        p = u[i][i] # pivot element
        for j in range(i+1, n):
            l[j][i] = u[j][i] / p
            u[j] = u[j] - l[j][i] * u[i]
    d = np.zeros(3)
    d = np.reshape(d, (3, 1))
    size_d = d.shape[0]
    # solve for Ly = b
    for i in range(size_d):
        sum = 0
        if i == 0:
            d[i] = result[i]
        else:
            for j in range(i):
                sum = sum + l[i][j] * d[j]
            d[i] = result[i] - sum
    # solve for UX = Y
    x = np.zeros(3)
    x = np.reshape(x, (3, 1))
    size_x = x.shape[0]
    for i in reversed(range(size_x)):
        sum = 0
        if i == size_x - 1:
            x[i] = d[i] / u[i][i]
        else:
            for j in range(i+1, size_x):
                sum = sum + (u[i][j] * x[j])
            x[i] = (d[i] - sum)/u[i][i]
    return x


def Gauss_Jordan(matrix, result):
    mat = np.array(matrix, dtype=float)
    res = np.array(result, dtype=float)
    res = np.reshape(res, (3, 1))
    augmented_matrix = np.hstack((mat, res))
    for i in range(augmented_matrix.shape[0]):
        scaled = False
        for j in range(augmented_matrix.shape[0]):
            if i != j:
                augmented_matrix[j] = augmented_matrix[j] - augmented_matrix[i] * augmented_matrix[j][i]
                if scaled is False and j != 0:
                    augmented_matrix[j] = augmented_matrix[j] / augmented_matrix[j][j]
                    scaled = True
    return augmented_matrix



def Gauss_Seidel(matrix, result, es):
    x = np.ones(3, dtype=float)  # initial guess
    ea = np.zeros(3, dtype=float)
    e_min = 1000  # random initial large value
    while e_min > es:
        for i in range(len(x)):
            sum = 0
            for j in range(len(x)):
                if i != j:
                    sum = sum + matrix[i][j]*x[j]
            old_x = x[i]
            x[i] = (result[i] - sum)/matrix[i][i]
            ea[i] = ((x[i] - old_x)/x[i]) * 100
        e_min = np.amax(ea)
    return x


test1_matrix = [[25.0, 5.0, 1.0], [64.0, 8.0, 1.0], [144.0, 12.0, 1.0]]
test1_result = [106.8, 177.2, 279.2]
test1 = Gauss_Elimination(test1_matrix, test1_result)
print(test1)

test2_matrix = [[1, 1, 2], [-1, -2, 3], [3, 7, 4]]
test2_result = [8, 1, 10]
test2 = Gauss_Jordan(test2_matrix, test2_result)
print(test2)

test3 = LU_Decomposition(np.matrix(test1_matrix), test1_result)
print(test3)

test4_matrix = [[12, 3, -5], [1, 5, 3], [3, 7, 13]]
test4_result = [1, 28, 76]
test4 = Gauss_Seidel(test4_matrix, test4_result, 0.1)
print(test4)
