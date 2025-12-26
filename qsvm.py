import matplotlib.pyplot as plt
import numpy as np
import os
import time

from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

if not os.path.exists("figures"):
    os.makedirs("figures")

def run_scenario(ax, noise_level, circuit_reps, scenario_name):
    print(f"Running scenario: {scenario_name} with noise level {noise_level} and circuit reps {circuit_reps}")

    X, y = make_moons(n_samples=100, noise=noise_level, random_state=42)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X= scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    feature_map = ZZFeatureMap(feature_dimension=2, reps=circuit_reps, entanglement='full')
    kernel = FidelityQuantumKernel(feature_map=feature_map)

    qsvc = QSVC(quantum_kernel=kernel)
    qsvc.fit(X_train, y_train)
    score = qsvc.score(X_test, y_test)

    h = 0.1
    x_min, x_max = X[: , 0].min() - 0.2, X[: , 0].max() + 0.2
    y_min, y_max = X[: , 1].min() - 0.2, X[: , 1].max() + 0.2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = qsvc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k', marker='o', label='Train')
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.coolwarm, edgecolors='k', marker='^', label='Test')

    ax.set_title(f"{scenario_name}\nNoise: {noise_level}, Reps: {circuit_reps}, Score: {score:.2f}")
    ax.set_xticks(())
    ax.set_yticks(())

def main_benchmark():
    print("Starting QSVM benchmark...")
    start_time = time.time()
    fig, axes = plt.subplots(2, 2, figsize=(14,10))
    fig.suptitle("QSVM Benchmarking Scenarios", fontsize=16)

    #scenario 1 : Easy and fast
    run_scenario(axes[0, 0], noise_level=0.05, circuit_reps=1, scenario_name="Underfitting Scenario (too simple)")

    #scenario 2 : Easy and complex
    run_scenario(axes[0, 1], noise_level=0.05, circuit_reps=4, scenario_name="Ideal (clean & deep)")

    #scenario 3 : difficult and fast
    run_scenario(axes[1, 0], noise_level=0.25, circuit_reps=1, scenario_name="Generaliation Challenge (noisy & shallow)")

    #scenario 4 : difficult and complex
    run_scenario(axes[1, 1], noise_level=0.25, circuit_reps=4, scenario_name="Overfitting Scenario (too complex)")

    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig_path = os.path.join("figures", "qsvm_benchmark.png")
    plt.savefig(fig_path)
    print(f"Benchmark completed in {time.time() - start_time:.2f} seconds.")
    print(f"Figure saved to {fig_path}")
    plt.show()

if __name__ == "__main__":
    main_benchmark()

