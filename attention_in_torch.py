import torch

def calcnorma(q_t, k_t, n, m):
    norma_q = torch.sqrt(torch.sum(q_t ** 2, dim=1))
    norma_k = torch.sqrt(torch.sum(k_t ** 2, dim=1))
    return norma_q, norma_k

def scale(q_t, k_t, norma_q, norma_k, n, m):
    q_t = q_t / norma_q.view(-1, 1) 
    k_t = k_t / norma_k.view(-1, 1)  
    return q_t, k_t

def print_matrix(matrix):
    for row in matrix:
        print(" ".join(f"{elem:.4f}" for elem in row))
    print()

def main():
    n = 2
    m = 3

    init_q = torch.tensor([[1.0, 2.0, 3.0], [4.0, 1.0, 0.0]])
    init_k = torch.tensor([[1.0, 0.0, 7.0], [2.0, 0.0, 1.0]])

    norma_q, norma_k = calcnorma(init_q, init_k, n, m)

    q_scaled, k_scaled = scale(init_q, init_k, norma_q, norma_k, n, m)

    print("Scaled matrix q:")
    print_matrix(q_scaled)

    print("\nScaled matrix k:")
    print_matrix(k_scaled)

if __name__ == "__main__":
    main()
