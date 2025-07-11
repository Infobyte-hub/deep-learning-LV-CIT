import pandas as pd
import os
import matplotlib.pyplot as plt
import glob

data_root = os.path.join(
    "data", "lvcit", "5res_analyse", "ca_gens"
)
ca_dir = os.path.join(
    "data", "lvcit", "1covering_array"
)


def load_ca_info():
    ca_time_size_df = pd.DataFrame(columns=["n", "k", "tau", "method", "size", "time"])
    ca_type_name = {
        "adaptive random": "LV-CIT",
        "baseline": "Baseline"
    }
    for ca_type in ["adaptive random", "baseline"]:
        for n in [20, 80]:
            for k in range(2, 7):
                tau = 2
                ca_path = glob.glob(os.path.join(
                    ca_dir, ca_type,
                    f"ca_{ca_type}_{n}_{k}_{tau}*.csv"
                ))
                for file in ca_path:
                    ca_time = float(file.split("_")[-1].replace(".csv", ""))
                    ca_size = int(file.split("_")[-2])
                    ca_time_size_df.loc[len(ca_time_size_df)] = [n, k, tau, ca_type_name[ca_type], ca_size, ca_time]
    acts_file = os.path.join(ca_dir, "ca_acts_info.csv")
    acts_df = pd.read_csv(acts_file)
    acts_df.sort_values(by=["n", "k", "i"], inplace=True)
    acts_df.drop(["i"], axis=1, inplace=True)
    acts_df["method"] = "ACTS"
    ca_time_size_df = pd.concat([ca_time_size_df, acts_df], axis=0).reset_index(drop=True)
    ca_time_size_df = ca_time_size_df.groupby(["n", "k", "tau", "method"]).agg({
        "size": "mean",
        "time": "mean"
    }).reset_index()
    ca_time_size_df.to_csv(os.path.join(data_root, "ca_info.csv"), index=False)
    return ca_time_size_df


def draw_ca_fig(ca_info):
    print(ca_info)
    methdos = ["Baseline", "LV-CIT", "ACTS"]
    markers = ["o", "s", "v"]
    colors = ["r", "g", "b"]
    line_styles = ["--", "-", "-."]
    y_label_names = {
        "size": "Size of Covering Array",
        "time": "Execution Time Cost (s)",
    }
    ns = [20, 80]

    title_font_size = 20
    index_font_size = 15

    for y_label in ["size", "time"]:
        for n in ns:
            tau = 2
            fig = plt.figure()
            ax = fig.add_subplot(111)
            x = ca_info[ca_info["n"] == n]["k"].drop_duplicates().values
            plt.xticks(fontsize=index_font_size)
            plt.yticks(fontsize=index_font_size)
            ax.set_xticks(x)
            # if y_label == "time":
            #     ax.set_yscale("log", base=10)
            for i in range(3):
                y = ca_info[(ca_info["n"] == n) & (ca_info["method"] == methdos[i]) & (ca_info["tau"] == tau)][y_label].values
                if not len(y):
                    continue
                ax.plot(x, y, label=methdos[i], marker=markers[i], color=colors[i], linestyle=line_styles[i])

            plt.legend(fontsize=title_font_size)
            plt.xlabel(r"Counting Constraint Variable $k$", fontsize=title_font_size)
            plt.ylabel(y_label_names[y_label], fontsize=title_font_size)
            # plt.show()
            fig.savefig(os.path.join(data_root, f"ca_{y_label}{n}_{tau}.png"), dpi=300, bbox_inches='tight')
            plt.close()
    return


if __name__ == "__main__":
    ca_info = load_ca_info()
    draw_ca_fig(ca_info)
