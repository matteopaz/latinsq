def min_entropy_latin_square(n):
    return [[(r + c) % n for c in range(n)] for r in range(n)]

def max_entropy_latin_square(n):
    s = [0]
    for k in range(1, n // 2):
        s.append(k)
        s.append(n - k)
    s.append(n // 2)
    
    n = len(s)
    return [[(c + s[r]) % n for c in range(n)] for r in range(n)]


def adjacency_matrix(L):
    n = len(L)
    A = [[0] * n for _ in range(n)]

    for r in range(n - 1):
        for c in range(n):
            i = L[r][c]
            j = L[r + 1][c]
            A[i][j] += 1

    return A