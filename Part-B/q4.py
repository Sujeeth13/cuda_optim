from matplotlib import pyplot as plt
import numpy as np

cpu_exec = [0.015,0.05,0.09,0.433,0.873]

no_unify_data_1x1 = [0.657,0.951,1.271,3.657,6.814]
no_unify_data_1x256 = [0.566,0.613,0.671,1.069,1.563]
no_unify_data_nx256 = [0.549,0.603,0.652,0.999,1.448]

unify_data_1x1 = [0.655,0.859,1.131,3.362,6.146]
unify_data_1x256 = [0.5,0.622,0.685,1.178,1.809]
unify_data_nx256 = [0.486,0.618,0.681,1.152,1.723]

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