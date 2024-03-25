from matplotlib import pyplot as plt
import numpy as np

cpu_exec = [0.00411296,0.01843,0.0351858,0.175048,0.360176]

no_unify_data_1x1 = [0.093455,0.397582,0.690609,2.802862,5.593631]
no_unify_data_1x256 = [0.004812,0.020981,0.038902,0.186475,0.372736]
no_unify_data_nx256 = [0.002864,0.009762,0.018726,0.089414,0.178199]

unify_data_1x1 = [0.090517,0.307462,0.538167,2.434372,4.802354]
unify_data_1x256 = [0.005115,0.023312,0.046180,0.224471,0.432334]
unify_data_nx256 = [0.004346,0.018968,0.037117,0.180375,0.371507]

k_values =[1,5,10,50,100]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(k_values, cpu_exec, label='CPU Execution', marker='o', linestyle='-')
plt.plot(k_values, no_unify_data_1x1, label='GPU No Unified 1x1', marker='s', linestyle='-')
plt.plot(k_values, no_unify_data_1x256, label='GPU No Unified 1x256', marker='^', linestyle='-')
plt.plot(k_values, no_unify_data_nx256, label='GPU No Unified Nx256', marker='*', linestyle='-')

plt.xlabel('K (in millions)')
plt.ylabel('Time to execute program (seconds)')
plt.title('Execution Time Comparison')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig('q4_without_unified.jpg', dpi=300) 

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(k_values, cpu_exec, label='CPU Execution', marker='o', linestyle='-')
plt.plot(k_values, unify_data_1x1, label='GPU Unified 1x1', marker='s', linestyle='-')
plt.plot(k_values, unify_data_1x256, label='GPU Unified 1x256', marker='^', linestyle='-')
plt.plot(k_values, unify_data_nx256, label='GPU Unified Nx256', marker='*', linestyle='-')

plt.xlabel('K (in millions)')
plt.ylabel('Time to execute program (seconds)')
plt.title('Execution Time Comparison')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig('q4_with_unified.jpg', dpi=300)