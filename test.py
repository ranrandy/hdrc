import os


if __name__ == "__main__":

    sep = "+" * 100

    # 1. Sample function f(x, y) = sin(pi/100*(x+y))
    # os.system("nvcc -O2 .\\poisson_solvers\\debug.cu .\\poisson_solvers\\debug_function2D.cu .\\poisson_solvers\\solvers.cu -o debug")
    for i in range(5):
        os.system(f"echo {sep}")
        os.system(f".\\poisson_solvers\\debug.exe {i} 0 1 10000 500 0.00001")
        os.system(f"nsys profile --output=.\\poisson_solvers\\gpu_profiling\\report_method_{i} .\\poisson_solvers\\debug {i} 5 20 10000 500 0.00001")
    # for i in range(5, 8): 
    #     # Because this sample function has the same frequency across the plane. Mutigrid methods won't actually help too much.
    #     os.system(f"echo {sep}")
    #     os.system(f".\\poisson_solvers\\debug.exe {i} 0 1 10 2 0.00001 4 10 300 10000 100 0.00001")
    #     os.system(f"nsys profile --output=.\\poisson_solvers\\gpu_profiling\\report_method_{i} .\\poisson_solvers\\debug {i} 5 20 10 2 0.00001 4 10 300 10000 100 0.00001")

    # 2. HDRC
    for fp in ["data\\belgium.hdr", "data\\bigFogMap.hdr"]:
        os.system(f"echo {sep}")
        os.system(f"echo {sep}")
        os.system(f"echo {sep}")
        os.system(f"echo {fp}")
        # CUDA
        for i in range(5):
            os.system(f"echo {sep}")
            os.system(f"python hdrc.py --source {fp} --save_att --cuda --method {i}")
        for i in range(5, 8):
            os.system(f"echo {sep}")
            os.system(f"python hdrc.py --source {fp} --save_att --cuda --method {i} --max_iterations 1 --check_frequency 10")
        # Python
        os.system(f"echo {sep}")
        os.system("python hdrc.py --save_att --method 0")