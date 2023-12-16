import matplotlib.pyplot as plt

# Plotting the data
plt.figure(figsize=(10, 6))

for m, method in zip(range(3), ["Jacobi", "Gauss-Seidel", "Gauss-Seidel + SoR"]):
    out_file = f"poisson_solvers\\out\\method_{m}.txt"
    # Parsing the data
    iterations = []
    residual_errors = []

    with open(out_file, 'r', encoding='utf-16') as file:
        for line in file:
            if line.startswith('Iteration:'):
                parts = line.split(',')
                iteration = int(parts[0].split(':')[1].strip())
                residual_error = float(parts[1].split(':')[1].strip())
                iterations.append(iteration)
                residual_errors.append(residual_error)

    plt.plot(iterations[100:], residual_errors[100:], linestyle='-', label=method)

plt.legend(loc='upper right', fontsize="25")

plt.title('Residual/Error vs Iteration Number')
plt.xlabel('Iteration Number')
plt.ylabel('Residual/Error')
plt.grid(True)
# plt.xscale('log')
# plt.yscale('log')
plt.savefig("error_fn.png", dpi=500)
plt.show()