import os
from string import Template
import subprocess
import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time


args_template = Template("""
[System]
Name: $name

[Parameter]
$parameters

[Constraint]
$constraint
""")


def generate_args():
    for n in [20, 80]:
        for k in range(2, 7):
            name = f"args_{n}_{k}"
            constraint = "+".join([f"P{i}" for i in range(1, n + 1)]) + f"<={k}"
            parameters = "\n".join([f"P{i} (int) : 0, 1" for i in range(1, n + 1)])
            args = args_template.substitute(name=name, parameters=parameters, constraint=constraint)
            with open(os.path.join("args", f"{name}.txt"), "w") as f:
                f.write(args)


complete_info = pd.DataFrame(columns=["k", "20-2", "80-2"])
complete_info["k"] = [2, 3, 4, 5, 6]
complete_info.fillna("x", inplace=True)


def update_complete_info(n, k, tau, i, state):
    if state == 0:
        if complete_info.loc[complete_info["k"] == k, f"{n}-{tau}"].str.contains("Running").any():
            complete_info.loc[complete_info["k"] == k, f"{n}-{tau}"] = \
                complete_info.loc[complete_info["k"] == k, f"{n}-{tau}"] + f", No.{i} by {threading.current_thread().name.replace('ThreadPoolExecutor-0_', '')}({time.strftime('%H:%M', time.localtime())})"
        else:
            complete_info.loc[complete_info["k"] == k, f"{n}-{tau}"] = f"Running No.{i} by {threading.current_thread().name.replace('ThreadPoolExecutor-0_', '')}({time.strftime('%H:%M', time.localtime())})"
    elif state == 1:
        if i == 5:
            complete_info.loc[complete_info["k"] == k, f"{n}-{tau}"] = "✓"
        elif complete_info.loc[complete_info["k"] == k, f"{n}-{tau}"].str == "✓":
            pass
        else:
            complete_info.loc[complete_info["k"] == k, f"{n}-{tau}"] = f"{i}/5"
    # os.system("cls")
    print(complete_info)


def run_acts(n, k, tau, i):
    # print(f"{threading.current_thread().name} Running ACTS for n={n}, k={k}, tau={tau}, No.{i}")
    update_complete_info(n, k, tau, i, 0)
    args_file = os.path.join("args", f"args_{n}_{k}.txt")
    output_file = os.path.join("res", f"ca_acts_{n}_{k}_{tau}_{i}.csv")
    command = ["java", f"-Ddoi={tau}", "-Doutput=csv", "-Dchandler=solver", "-Dprogress=on", "-Drandstar=on", "-jar", "acts_3.2.jar", args_file, output_file]
    process = subprocess.run(command, capture_output=True, text=True)
    ca_size, ca_time = 0, 0
    if process.returncode == 0:
        output = process.stdout.strip()
        if output:
            with open(os.path.join("logs", f"ca_acts_{n}_{k}_{tau}_{i}.log"), "w") as f:
                f.write(output)
            info = re.search(r"Number of Tests\t: (\d+)\nTime \(seconds\)\t: ([\d\.]+)", output)
            if info:
                ca_size = info.group(1)
                ca_time = info.group(2)
        else:
            error = process.stderr.strip()
            if error:
                with open(os.path.join("logs", f"ca_acts_{n}_{k}_{tau}_{i}-err.log"), "w") as f:
                    f.write(error)
    else:
        error = process.stderr.strip()
        if error:
            with open(os.path.join("logs", f"ca_acts_{n}_{k}_{tau}_{i}-err.log"), "w") as f:
                f.write(error)
    # print(f"{threading.current_thread().name} Finished ACTS for n={n}, k={k}, tau={tau}, No.{i}, size={ca_size}, time={ca_time}")
    update_complete_info(n, k, tau, i, 1)
    return n, k, tau, i, ca_size, ca_time


def runner(max_workers=10):
    df = pd.DataFrame(columns=["n", "k", "tau", "i", "size", "time"])
    thread_pool = ThreadPoolExecutor(max_workers=max_workers)
    res = []
    for n in [20, 80]:
        for i in range(5):
            for k in range(2, 7):
                tau = 2
                res.append(thread_pool.submit(run_acts, n, k, tau, i + 1))
    for future in as_completed(res):
        n, k, tau, i, ca_size, ca_time = future.result()
        df.loc[len(df)] = [n, k, tau, i, ca_size, ca_time]
        df.to_csv("ca_acts_info_tmp.csv", index=False)
    df.sort_values(by=["n", "k", "tau", "i"], inplace=True)
    df.to_csv("ca_acts_info.csv", index=False)


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('max_colwidth', 1000)
    pd.set_option('display.width', 1000)
    generate_args()
    if not os.path.exists("args"):
        os.makedirs("args")
    if not os.path.exists("res"):
        os.makedirs("res")
    runner(10)
