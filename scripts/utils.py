import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# a function that takes the miss ratio of all traces and plot the relative miss ratio
def plot_general(
    data_array=None,
    xticks=None,
    yticks=None,
    xlabel=None,
    ylabel=None,
    name=None,
    yy=0,
    yx=0,
    **kwargs,
):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    box = ax.boxplot(
        data_array,
        patch_artist=True,
        showfliers=False,
        whis=[10, 90],
        showmeans=True,
        meanprops=dict(
            marker="o", markerfacecolor="white", markeredgecolor="black", markersize=8
        ),
        medianprops=dict(linestyle="-", linewidth=1.25, color="black"),
    )
    for box_item in box["boxes"]:
        box_item.set(color="black", linewidth=1.25)  # Edge color and line width
        box_item.set(facecolor="lightblue")  # Fill color
    box["boxes"][-1].set_facecolor("lightgreen")
    plt.grid(True, color="lightgray", linestyle="--", linewidth=1)
    plt.xlabel(xlabel, fontsize=19)
    plt.xticks(np.arange(1, len(xticks) + 1), xticks, fontsize=18)
    plt.ylabel(ylabel, fontsize=19)
    plt.yticks(fontsize=18)
    if yticks is not None:
        print("here")
        plt.yticks(yticks)

    ax.yaxis.set_label_coords(
        ax.yaxis.get_label().get_position()[0] - yx,
        ax.yaxis.get_label().get_position()[1] - yy,
    )  # Adjust the y-coordinate
    plt.savefig("plots/" + name + ".pdf", bbox_inches="tight", format="pdf")
    plt.close()


def plot_rel_miss(
    ALGO_miss_ratio,
    FIFO_miss_ratio,
    LRU_miss_ratio,
    ALGO_params,
    color_diff=1,
    **kwargs,
):
    # rel. to LRU
    ALGO_relative_miss_stack = ALGO_miss_ratio / LRU_miss_ratio

    fig, ax = plt.subplots(figsize=(6, 3.8))
    box = ax.boxplot(
        ALGO_relative_miss_stack,
        patch_artist=True,
        showfliers=False,
        whis=[10, 90],
        showmeans=True,
        meanprops=dict(
            marker="o", markerfacecolor="white", markeredgecolor="black", markersize=8
        ),
        medianprops=dict(linestyle="-", linewidth=1.25, color="black"),
    )
    for i, box_values in enumerate(box["medians"]):
        median = box_values.get_ydata()[0]
        mean = box["means"][i].get_ydata()[0]  # If `showmeans=True`
        whisker_low = box["whiskers"][2 * i].get_ydata()[1]  # Lower whisker
        whisker_high = box["whiskers"][2 * i + 1].get_ydata()[1]  # Upper whisker
        # q1 = box['boxes'][i].get_ydata()[0]  # First quartile (lower edge of box)
        # q3 = box['boxes'][i].get_ydata()[2]  # Third quartile (upper edge of box)
        # print("cache size", kwargs.get("size_cache"))
        # print(f"Box {i+1}:")
        # print(f"  Median: {median}")
        # print(f"  Mean: {mean}")
        # print(f"  Lower Whisker (10th percentile): {whisker_low}")
        # print(f"  Upper Whisker (90th percentile): {whisker_high}")
        # print(f"  Q1 (25th percentile): {q1}")
        # print(f"  Q3 (75th percentile): {q3}")
    len_params = len(ALGO_params)
    for box_item in box["boxes"]:
        box_item.set(color="black", linewidth=1.25)  # Edge color and line width
        box_item.set(facecolor="lightblue")  # Fill color
    if kwargs.get("ALGO_name") == "beladyclock":
        box["boxes"][-1].set_facecolor("lightgreen")
    # make the last box different, no need if rel. to lru
    plt.grid(True, color="lightgray", linestyle="--", linewidth=1)
    plt.xlabel(kwargs.get("xlabel"), fontsize=19)
    plt.xticks(np.arange(1, len_params + 1), ALGO_params, fontsize=18)
    # plt.ylabel("Miss ratio relative to FIFO", fontsize=19)
    plt.ylabel("Miss ratio relative to LRU", fontsize=19)
    plt.yticks(fontsize=18)
    if kwargs.get("ALGO_name") == "clock":
        plt.ylim(0.93, 1.14)
        plt.yticks(np.arange(0.94, 1.15, 0.02), fontsize=18)
    elif (
        kwargs.get("ALGO_name") == "random_belady"
        or kwargs.get("ALGO_name") == "random_lru"
    ):
        plt.ylim(0.7, 1.1)
        plt.yticks(np.arange(0.7, 1.15, 0.05), fontsize=18)
    elif (
        kwargs.get("ALGO_name") == "hot_lru"
        or kwargs.get("ALGO_name") == "beladyclock"
        or kwargs.get("ALGO_name") == "random"
    ):
        pass
    else:
        plt.ylim(0.99, 1.14)
        plt.yticks(np.arange(1, 1.15, 0.02), fontsize=18)
    ax.yaxis.set_label_coords(
        ax.yaxis.get_label().get_position()[0] - 0.15,
        ax.yaxis.get_label().get_position()[1] - 0.05,
    )  # Adjust the y-coordinate
    plt.savefig(
        kwargs.get("ALGO_name") + "_miss_" + kwargs.get("size_cache") + ".pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_rel_random_belady_distribution(
    ALGO_relative_stack_rev, ALGO_params, color_diff=1, **kwargs
):
    # rel. to LRU
    ALGO_relative_stack = 1 - ALGO_relative_stack_rev

    fig, ax = plt.subplots(figsize=(6, 3.8))
    box = ax.boxplot(
        ALGO_relative_stack,
        patch_artist=True,
        showfliers=False,
        whis=[10, 90],
        showmeans=True,
        meanprops=dict(
            marker="o", markerfacecolor="white", markeredgecolor="black", markersize=8
        ),
        medianprops=dict(linestyle="-", linewidth=1.25, color="black"),
    )
    for i, box_values in enumerate(box["medians"]):
        median = box_values.get_ydata()[0]
        mean = box["means"][i].get_ydata()[0]  # If `showmeans=True`
        whisker_low = box["whiskers"][2 * i].get_ydata()[1]  # Lower whisker
        whisker_high = box["whiskers"][2 * i + 1].get_ydata()[1]  # Upper whisker
        # print("distribution")
        # print("cache size", kwargs.get("size_cache"))
        # print(f"Box {i+1}:")
        # print(f"  Median: {median}")
        # print(f"  Mean: {mean}")
        # print(f"  Lower Whisker (10th percentile): {whisker_low}")
        # print(f"  Upper Whisker (90th percentile): {whisker_high}")
    len_params = len(ALGO_params)
    ALGO_params[-1] = float("inf")
    for box_item in box["boxes"]:
        box_item.set(color="black", linewidth=1.25)  # Edge color and line width
        box_item.set(facecolor="lightblue")  # Fill color
    # make the last box different, no need if rel. to lru
    plt.grid(True, color="lightgray", linestyle="--", linewidth=1)
    plt.xlabel(kwargs.get("xlabel"), fontsize=19)
    plt.xticks(np.arange(1, len_params + 1), ALGO_params, fontsize=18)
    # plt.ylabel("Miss ratio relative to FIFO", fontsize=19)
    ax.set_ylabel("Fraction of evictions by Belady", fontsize=18)
    ax.yaxis.set_label_coords(
        ax.yaxis.get_label().get_position()[0] - 0.12,
        ax.yaxis.get_label().get_position()[1] - 0.05,
    )  # Adjust the y-coordinate
    plt.yticks(fontsize=18)
    print("algo name", kwargs.get("ALGO_name"))
    plt.savefig(
        kwargs.get("ALGO_name") + "_distribution_" + kwargs.get("size_cache") + ".pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_rel_promotions(ALGO_promotions, LRU_promotions, ALGO_params, **kwargs):
    LRU_promotions = np.tile(LRU_promotions, ALGO_promotions.shape[1])
    ALGO_relative_promotions = ALGO_promotions / LRU_promotions

    fig, ax = plt.subplots(figsize=(6, 3.8))
    box = ax.boxplot(
        ALGO_relative_promotions,
        patch_artist=True,
        showfliers=False,
        whis=[10, 90],
        showmeans=True,
        meanprops=dict(
            marker="o", markerfacecolor="white", markeredgecolor="black", markersize=8
        ),
        medianprops=dict(linestyle="-", linewidth=1.25, color="black"),
    )

    for box_item in box["boxes"]:
        box_item.set(color="black", linewidth=1.25)
        box_item.set(facecolor="lightblue")
    if kwargs.get("ALGO_name") == "beladyclock":
        box["boxes"][-1].set_facecolor("lightgreen")
    len_params = len(ALGO_params)
    plt.grid(True, color="lightgray", linestyle="--", linewidth=1)
    print("xlabel", kwargs.get("xlabel"))
    plt.xlabel(kwargs.get("xlabel"), fontsize=19)
    plt.xticks(np.arange(1, len_params + 1), ALGO_params, fontsize=18)
    plt.ylabel("Promotions relative to LRU", fontsize=19)
    plt.yticks(np.arange(0, 1.1, 0.2), fontsize=18)
    plt.ylim(0, 1)
    ax.yaxis.set_label_coords(
        ax.yaxis.get_label().get_position()[0] - 0.12,
        ax.yaxis.get_label().get_position()[1] - 0.05,
    )  # Adjust the y-coordinate
    plt.savefig(
        kwargs.get("ALGO_name")
        + "_promotion_boxplot_"
        + kwargs.get("size_cache")
        + ".pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_rel_promotions_comp(ALGO_promotions, LRU_promotions, ALGO_params, **kwargs):
    LRU_promotions = np.tile(LRU_promotions, ALGO_promotions.shape[1])
    ALGO_relative_promotions = ALGO_promotions / LRU_promotions
    orig_ALGO_relative_promotions = kwargs.get("orig_promotion") / LRU_promotions

    fig, ax = plt.subplots(figsize=(6, 3.8))
    box = ax.boxplot(
        np.array(
            [
                ALGO_relative_promotions[:, 0],
                ALGO_relative_promotions[:, 1],
                ALGO_relative_promotions[:, 2],
                orig_ALGO_relative_promotions[:, 0],
            ]
        ).T,
        patch_artist=True,
        showfliers=False,
        whis=[10, 90],
        showmeans=True,
        meanprops=dict(
            marker="o", markerfacecolor="white", markeredgecolor="black", markersize=8
        ),
        medianprops=dict(linestyle="-", linewidth=1.25, color="black"),
    )

    for box_item in box["boxes"]:
        box_item.set(color="black", linewidth=1.25)
        box_item.set(facecolor="lightblue")
    len_params = len(ALGO_params)
    plt.grid(True, color="lightgray", linestyle="--", linewidth=1)
    # plt.xlabel("Delay time", fontsize=19)
    plt.xlabel("Admission Threshold", fontsize=19)
    plt.xticks(np.arange(1, len_params + 1), ALGO_params, fontsize=18)
    plt.ylabel("Promotions relative to LRU", fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig(
        kwargs.get("ALGO_name")
        + "_promotion_boxplot_"
        + kwargs.get("size_cache")
        + ".pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_promotion_efficiency(
    ALGO_promotions,
    LRU_promotions,
    ALGO_miss_ratio,
    FIFO_miss_ratio,
    LRU_miss_ratio,
    ALGO_params,
    TRACE_request,
    **kwargs,
):
    # print("get algo name", kwargs.get("ALGO_name"))
    # LRU_hit_ratio = 1 - LRU_miss_ratio
    # FIFO_hit_ratio = 1 - FIFO_miss_ratio

    LRU_misses = LRU_miss_ratio * TRACE_request
    FIFO_misses = FIFO_miss_ratio * TRACE_request

    # expand the FIFO_hit to the same shape as ALGO_promotions
    FIFO_misses = np.tile(FIFO_misses, ALGO_promotions.shape[1])
    LRU_misses = np.tile(LRU_misses, ALGO_promotions.shape[1])
    LRU_promotions = np.tile(LRU_promotions, ALGO_promotions.shape[1])
    ALGO_misses = ALGO_miss_ratio * TRACE_request
    miss_difference = (ALGO_miss_ratio - LRU_miss_ratio) / LRU_miss_ratio
    a = miss_difference
    b = (LRU_promotions - ALGO_promotions) / LRU_promotions
    row_mask_a = np.all(a > 0, axis=1)  # Rows where any element in 'a' > 0
    row_mask_b = np.all(b > 0, axis=1)  # Rows where any element in 'b' > 0

    # Combine the masks: Keep rows that satisfy both conditions
    combined_mask = row_mask_a & row_mask_b  # Logical AND of the two masks

    # Apply the combined mask to both arrays
    a_filtered = a[combined_mask]
    b_filtered = b[combined_mask]
    ALGO_promotion_efficiency = b_filtered / a_filtered
    mean_bar = np.mean(ALGO_promotion_efficiency, axis=0)

    mean_miss_ratio_algo = np.mean(ALGO_miss_ratio, axis=0)
    mean_miss_ratio_lru = np.mean(LRU_miss_ratio, axis=0)
    mean_promotion_algo = np.mean(ALGO_promotions, axis=0)
    mean_promotion_lru = np.mean(LRU_promotions, axis=0)
    mean_efficiency_manual = (
        (mean_promotion_lru - mean_promotion_algo) / mean_promotion_lru
    ) / ((mean_miss_ratio_algo - mean_miss_ratio_lru) / mean_miss_ratio_lru)
    print("algo: ", kwargs.get("ALGO_name"))
    print("miss ratio algo: ", mean_miss_ratio_algo)
    print("miss ratio lru: ", mean_miss_ratio_lru)
    print("promotion algo: ", mean_promotion_algo)
    print("promotion lru: ", mean_promotion_lru)
    print("efficiency manual: ", mean_efficiency_manual)
    print("mean promotion efficiency", mean_bar)

    # threshold = 10  # Define your threshold
    # mask = np.all((ALGO_promotion_efficiency > -threshold) & (ALGO_promotion_efficiency < threshold), axis=1)
    # sifted_count = np.sum(~mask)  # Count rows where the mask is False
    # print(f"Number of rows removed: {sifted_count}")

    # Keep only rows where the mask is True
    # ALGO_promotion_efficiency = ALGO_promotion_efficiency[mask]

    # LRU_promotion_efficiency_stack = np.hstack((LRU_promotion_efficiency, ALGO_promotion_efficiency))

    fig, ax = plt.subplots(figsize=(6, 3.8))
    box = ax.boxplot(
        ALGO_promotion_efficiency,
        patch_artist=True,
        showfliers=False,
        whis=[10, 90],
        showmeans=True,
        meanprops=dict(
            marker="o", markerfacecolor="white", markeredgecolor="black", markersize=8
        ),
        medianprops=dict(linestyle="-", linewidth=1.25, color="black"),
    )

    for i, box_item in enumerate(box["boxes"]):
        box_item.set(color="black", linewidth=1.25)
        box_item.set(facecolor="lightblue")
        # if "hot_lru" in kwargs.get("ALGO_name") :
        #     mean = box['means'][i].get_ydata()[0]  # If `showmeans=True`
        #     print("mean promotion efficiency", mean)

    box["boxes"][-1].set_facecolor("lightgreen")
    # box['boxes'][-2].set_hatch("xxx")

    num_params = len(ALGO_params)
    plt.grid(True, color="lightgray", linestyle="--", linewidth=1)
    plt.xlabel(kwargs.get("xlabel"), fontsize=19)
    plt.xticks(np.arange(1, num_params + 1), ALGO_params, fontsize=18)
    plt.ylabel("placeholder2", fontsize=18)
    # ax.yaxis.set_label_coords(ax.yaxis.get_label().get_position()[0] - 0.2, ax.yaxis.get_label().get_position()[1] -0.05)  # Adjust the y-coordinate
    plt.yticks(fontsize=18)
    # if "hot_lru" in kwargs.get("ALGO_name") :
    #     plt.ylim(-0.5, 2.5)
    # elif not "beladyclock" in kwargs.get("ALGO_name"):
    #     plt.ylim(-0.09,0.2)
    #     plt.yticks(np.arange(-0.12, 0.21, 0.04), fontsize=18)
    plt.savefig(
        kwargs.get("ALGO_name")
        + "_efficiency_boxplot_"
        + kwargs.get("size_cache")
        + ".pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_customized_promotion_efficiency(
    ALGOs_promotions,
    LRU_promotions,
    ALGOs_miss_ratio,
    FIFO_miss_ratio,
    LRU_miss_ratio,
    ALGOs_params,
    TRACE_request,
    **kwargs,
):
    LRU_hit_ratio = 1 - LRU_miss_ratio
    FIFO_hit_ratio = 1 - FIFO_miss_ratio

    LRU_hit = LRU_hit_ratio * TRACE_request
    FIFO_hit = FIFO_hit_ratio * TRACE_request

    LRU_promotion_efficiency = (LRU_hit - FIFO_hit) / LRU_promotions

    LRU_promotion_efficiency_stack = None
    # expand the FIFO_hit to the same shape as ALGO_promotions
    print("ALGOs_promotions shape", ALGOs_promotions.shape)
    FIFO_hit = np.tile(FIFO_hit, ALGOs_promotions.shape[2])
    for i in range(ALGOs_promotions.shape[0]):
        ALGO_promotions = ALGOs_promotions[i]
        ALGO_miss_ratio = ALGOs_miss_ratio[i]
        ALGO_hit_ratio = 1 - ALGO_miss_ratio
        # print("hit raio shape", ALGO_hit_ratio.shape)
        ALGO_hit = ALGO_hit_ratio * TRACE_request
        # print("algo hit: ", ALGO_hit)
        # print("fifo hit: ", FIFO_hit)
        ALGO_promotion_efficiency = (ALGO_hit - FIFO_hit) / ALGO_promotions
        print("i", i)
        print("algo promotion efficiency", ALGO_promotion_efficiency)
        LRU_promotion_efficiency_stack = (
            ALGO_promotion_efficiency
            if LRU_promotion_efficiency_stack is None
            else np.hstack((LRU_promotion_efficiency_stack, ALGO_promotion_efficiency))
        )

    ALGOs_params = ALGOs_params + ["LRU"]
    LRU_promotion_efficiency_stack = np.hstack(
        (LRU_promotion_efficiency_stack, LRU_promotion_efficiency)
    )

    fig, ax = plt.subplots(figsize=(6, 3.8))
    box = ax.boxplot(
        LRU_promotion_efficiency_stack,
        patch_artist=True,
        showfliers=False,
        whis=[10, 90],
        showmeans=True,
        meanprops=dict(
            marker="o", markerfacecolor="white", markeredgecolor="black", markersize=8
        ),
        medianprops=dict(linestyle="-", linewidth=1.25, color="black"),
    )

    for box_item in box["boxes"]:
        box_item.set(color="black", linewidth=1.25)
        box_item.set(facecolor="lightblue")

    box["boxes"][-1].set_facecolor("lightpink")

    num_params = len(ALGOs_params)
    plt.grid(True, color="lightgray", linestyle="--", linewidth=1)
    plt.xlabel(kwargs.get("xlabel"), fontsize=19)
    plt.xticks(np.arange(1, num_params + 1), ALGOs_params, fontsize=18)
    plt.ylabel("Promotion efficiency", fontsize=19)
    plt.yticks(fontsize=18)
    plt.savefig(
        kwargs.get("title")
        + "_efficiency_boxplot_"
        + kwargs.get("size_cache")
        + ".pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_multi_rel_miss(
    ALGOs_miss_ratio, FIFO_miss_ratio, LRU_miss_ratio, ALGOs_params, **kwargs
):
    # print("fifo miss ratio", FIFO_miss_ratio)
    LRU_relative_miss = LRU_miss_ratio / FIFO_miss_ratio

    ALGO_relative_miss_stack = None
    for i in range(ALGOs_miss_ratio.shape[0]):
        ALGO_miss = ALGOs_miss_ratio[i]
        FIFO_miss_ratio_variant = np.tile(FIFO_miss_ratio, ALGO_miss.shape[1])
        ALGO_relative_miss = ALGO_miss / FIFO_miss_ratio_variant
        ALGO_relative_miss_stack = (
            ALGO_relative_miss
            if ALGO_relative_miss_stack is None
            else np.hstack((ALGO_relative_miss_stack, ALGO_relative_miss))
        )

    ALGO_relative_miss_stack = np.hstack((ALGO_relative_miss_stack, LRU_relative_miss))
    ALGOs_params = ALGOs_params + ["LRU"]

    # adjust the positions
    total_boxes = ALGO_relative_miss_stack.shape[1]
    positions = np.arange(1, total_boxes + 1)
    shift = 0.15  # Controls how close the pairs are
    adjusted_positions = []
    for i in range(len(ALGOs_params) - 1):  # Group pairs, excluding the last "LRU"
        adjusted_positions.append(positions[i] - shift)
        adjusted_positions.append(positions[i] + shift)
    adjusted_positions.append(positions[-1])
    print("adjusted_positions:", adjusted_positions)

    fig, ax = plt.subplots(figsize=(6, 3.8))
    box = ax.boxplot(
        ALGO_relative_miss_stack,
        positions=adjusted_positions,
        patch_artist=True,
        showfliers=False,
        whis=[10, 90],
        showmeans=True,
        meanprops=dict(
            marker="o", markerfacecolor="white", markeredgecolor="black", markersize=8
        ),
        medianprops=dict(linestyle="-", linewidth=1.25, color="black"),
    )
    for i, box_values in enumerate(box["medians"]):
        median = box_values.get_ydata()[0]
        mean = box["means"][i].get_ydata()[0]  # If `showmeans=True`
        whisker_low = box["whiskers"][2 * i].get_ydata()[1]  # Lower whisker
        whisker_high = box["whiskers"][2 * i + 1].get_ydata()[1]  # Upper whisker
        # q1 = box['boxes'][i].get_ydata()[0]  # First quartile (lower edge of box)
        # q3 = box['boxes'][i].get_ydata()[2]  # Third quartile (upper edge of box)
        print("cache size", kwargs.get("size_cache"))
        print(f"Box {i + 1}:")
        print(f"  Median: {median}")
        print(f"  Mean: {mean}")
        print(f"  Lower Whisker (10th percentile): {whisker_low}")
        print(f"  Upper Whisker (90th percentile): {whisker_high}")
        # print(f"  Q1 (25th percentile): {q1}")
        # print(f"  Q3 (75th percentile): {q3}")
    len_params = len(ALGOs_params)
    for box_item in box["boxes"]:
        box_item.set(color="black", linewidth=1.25)  # Edge color and line width
        box_item.set(facecolor="lightblue")  # Fill color
    # make the last box different
    box["boxes"][-1].set_facecolor("lightpink")
    box["boxes"][-1].set_hatch("///")
    plt.grid(True, color="lightgray", linestyle="--", linewidth=1)
    plt.xlabel(kwargs.get("xlabel"), fontsize=19)
    plt.xticks(np.arange(1, len_params + 1), ALGOs_params, fontsize=18)
    plt.ylabel("Miss ratio relative to FIFO", fontsize=19)
    plt.yticks(fontsize=18)
    plt.savefig(
        kwargs.get("title") + "_boxplot_" + kwargs.get("size_cache") + ".pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_compare_rel_miss(
    ALGO1s_miss_ratio, ALGO2s_miss_ratio, LRU_miss_ratio, params, **kwargs
):
    # interleaving the ALGOi_miss_ratio and ALGO2_miss_ratio
    ALGO_miss_ratio_stack = None
    # print("ALGO1s_miss_ratio shape", ALGO1s_miss_ratio.shape[1])
    for i in range(ALGO2s_miss_ratio.shape[1]):
        ALGO1_miss_ratio = ALGO1s_miss_ratio[:, i]
        ALGO1_miss_ratio = np.reshape(ALGO1_miss_ratio, (ALGO1_miss_ratio.shape[0], 1))
        ALGO2_miss_ratio = ALGO2s_miss_ratio[:, i]
        ALGO2_miss_ratio = np.reshape(ALGO2_miss_ratio, (ALGO2_miss_ratio.shape[0], 1))
        ALGO_miss_ratio = ALGO1_miss_ratio / LRU_miss_ratio
        ALGO2_miss_ratio = ALGO2_miss_ratio / LRU_miss_ratio
        if ALGO_miss_ratio_stack is None:
            ALGO_miss_ratio_stack = np.hstack((ALGO_miss_ratio, ALGO2_miss_ratio))
        else:
            ALGO_miss_ratio_stack = np.hstack((ALGO_miss_ratio_stack, ALGO_miss_ratio))
            ALGO_miss_ratio_stack = np.hstack((ALGO_miss_ratio_stack, ALGO2_miss_ratio))
    fig, ax = plt.subplots(figsize=(6, 3.8))

    # adjust the positions
    total_boxes = ALGO_miss_ratio_stack.shape[1]
    positions = np.arange(1, total_boxes + 1)
    adjusted_positions = []
    print("inttotal_boxes", int((total_boxes) / 2))
    interval = 0.15
    for i in range(int((total_boxes) / 2)):  # Group pairs, excluding the last "LRU"
        if i > 2:
            shift = 1
        else:
            shift = 0
        adjusted_positions.append(positions[2 * i] + interval + shift)
        adjusted_positions.append(positions[2 * i + 1] - interval + shift)
    # print("adjusted_positions:", adjusted_positions)

    box = ax.boxplot(
        ALGO_miss_ratio_stack,
        positions=adjusted_positions,
        patch_artist=True,
        showfliers=False,
        whis=[10, 90],
        showmeans=True,
        meanprops=dict(
            marker="o", markerfacecolor="white", markeredgecolor="black", markersize=8
        ),
        medianprops=dict(linestyle="-", linewidth=1.25, color="black"),
    )
    len_params = len(params) * 2
    for i, box_values in enumerate(box["boxes"]):
        if i % 2 == 1:
            # use lightpink
            box_values.set(color="black", linewidth=1.25)  # Edge color and line width
            box_values.set(facecolor="lightpink")
            box_values.set_hatch("///")
        else:
            box_values.set(color="black", linewidth=1.25)  # Edge color and line width
            box_values.set(facecolor="lightblue")  # Fill color

    plt.grid(True, color="lightgray", linestyle="--", linewidth=1)
    plt.xlabel(kwargs.get("xlabel"), fontsize=19)
    group_centers = np.arange(1, total_boxes // 2 + 1) * 2 - 0.5
    group_centers[-1] = group_centers[-1] + 1
    # plt.xticks(np.arange(1, len_params + 1), fontsize=18)
    plt.xticks(group_centers, params, fontsize=18)
    plt.ylabel("Miss ratio relative to LRU", fontsize=19)
    plt.yticks(fontsize=18)
    ax.yaxis.set_label_coords(
        ax.yaxis.get_label().get_position()[0] - 0.15,
        ax.yaxis.get_label().get_position()[1] - 0.05,
    )  # Adjust the y-coordinate
    plt.savefig(
        kwargs.get("ALGO_name") + "_boxplot_" + kwargs.get("size_cache") + ".pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_compare_rel_promotions(
    ALGO1s_promotions, ALGO2s_promotions, LRU_promotions, ALGOs_params, **kwargs
):
    # LRU_promotions = np.tile(LRU_promotions, ALGO1s_promotions.shape[1])
    ALGO_relative_promotions_stack = None
    for i in range(ALGO1s_promotions.shape[1]):
        ALGO1_relative_promotion = ALGO1s_promotions[:, i]
        ALGO2_relative_promotion = ALGO2s_promotions[:, i]
        ALGO1_relative_promotion = np.reshape(
            ALGO1_relative_promotion, (ALGO1_relative_promotion.shape[0], 1)
        )
        ALGO1_relative_promotion = ALGO1_relative_promotion / LRU_promotions
        ALGO2_relative_promotion = np.reshape(
            ALGO2_relative_promotion, (ALGO2_relative_promotion.shape[0], 1)
        )
        ALGO2_relative_promotion = ALGO2_relative_promotion / LRU_promotions
        if ALGO_relative_promotions_stack is None:
            ALGO_relative_promotions_stack = np.hstack(
                (ALGO1_relative_promotion, ALGO2_relative_promotion)
            )
        else:
            ALGO_relative_promotions_stack = np.hstack(
                (ALGO_relative_promotions_stack, ALGO1_relative_promotion)
            )
            ALGO_relative_promotions_stack = np.hstack(
                (ALGO_relative_promotions_stack, ALGO2_relative_promotion)
            )

    # adjust the positions
    total_boxes = ALGO_relative_promotions_stack.shape[1]
    positions = np.arange(1, total_boxes + 1)
    interval = 0.15  # Controls how close the pairs are
    adjusted_positions = []
    for i in range(int((total_boxes) / 2)):  # Group pairs, excluding the last "LRU"
        if i > 2:
            shift = 1
        else:
            shift = 0
        adjusted_positions.append(positions[2 * i] + interval + shift)
        adjusted_positions.append(positions[2 * i + 1] - interval + shift)
    # print("adjusted_positions:", adjusted_positions)

    fig, ax = plt.subplots(figsize=(6, 3.8))
    box = ax.boxplot(
        ALGO_relative_promotions_stack,
        positions=adjusted_positions,
        patch_artist=True,
        showfliers=False,
        whis=[10, 90],
        showmeans=True,
        meanprops=dict(
            marker="o", markerfacecolor="white", markeredgecolor="black", markersize=8
        ),
        medianprops=dict(linestyle="-", linewidth=1.25, color="black"),
    )

    for i, box_values in enumerate(box["boxes"]):
        if i % 2 == 1:
            # use lightpink
            box_values.set(color="black", linewidth=1.25)
            box_values.set(facecolor="lightpink")
            box_values.set_hatch("///")
        else:
            box_values.set(color="black", linewidth=1.25)
            box_values.set(facecolor="lightblue")

    lightpink_legend = mpatches.Patch(
        facecolor="lightpink", edgecolor="black", hatch="///", label="HotCache/AGE"
    )
    lightblue_legend = mpatches.Patch(
        facecolor="lightblue", edgecolor="black", label="Original"
    )

    ax.legend(
        handles=[lightpink_legend, lightblue_legend], fontsize=18, loc="upper left"
    )

    len_params = len(ALGOs_params) * 2
    plt.grid(True, color="lightgray", linestyle="--", linewidth=1)
    plt.xlabel(kwargs.get("xlabel"), fontsize=19)
    group_centers = np.arange(1, total_boxes // 2 + 1) * 2 - 0.5
    group_centers[-1] = group_centers[-1] + 1
    # plt.xticks(np.arange(1, len_params + 1), fontsize=18)
    plt.xticks(group_centers, ALGOs_params, fontsize=18)
    plt.ylabel("Promotions relative to LRU", fontsize=19)
    plt.yticks(fontsize=18)
    ax.yaxis.set_label_coords(
        ax.yaxis.get_label().get_position()[0] - 0.12,
        ax.yaxis.get_label().get_position()[1] - 0.1,
    )  # Adjust the y-coordinate
    plt.savefig(
        kwargs.get("ALGO_name")
        + "_promotion_boxplot_"
        + kwargs.get("size_cache")
        + ".pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_compare_promotion_efficiency(
    ALGO1s_promotions,
    ALGO2s_promotions,
    LRU_promotions,
    ALGO1s_miss_ratio,
    ALGO2s_miss_ratio,
    FIFO_miss_ratio,
    LRU_miss_ratio,
    ALGOs_params,
    TRACE_request,
    **kwargs,
):
    LRU_misses = LRU_miss_ratio * TRACE_request
    FIFO_misses = FIFO_miss_ratio * TRACE_request

    LRU_promotion_efficiency_stack = None
    for i in range(ALGO1s_promotions.shape[1]):
        ALGO1_promotions = ALGO1s_promotions[:, i]
        ALGO1_promotions = np.reshape(ALGO1_promotions, (ALGO1_promotions.shape[0], 1))
        ALGO2_promotions = ALGO2s_promotions[:, i]
        ALGO2_promotions = np.reshape(ALGO2_promotions, (ALGO2_promotions.shape[0], 1))
        ALGO1_miss_ratio = ALGO1s_miss_ratio[:, i]
        ALGO1_miss_ratio = np.reshape(ALGO1_miss_ratio, (ALGO1_miss_ratio.shape[0], 1))
        ALGO2_miss_ratio = ALGO2s_miss_ratio[:, i]
        ALGO2_miss_ratio = np.reshape(ALGO2_miss_ratio, (ALGO2_miss_ratio.shape[0], 1))
        ALGO1_misses = ALGO1_miss_ratio * TRACE_request
        ALGO2_misses = ALGO2_miss_ratio * TRACE_request
        ALGO1_promotion_efficiency = (ALGO1_misses - LRU_misses) / (
            LRU_promotions - ALGO1_promotions
        )
        # print all promotion efficiency greater than 10
        outlier_indices = np.where(ALGO1_promotion_efficiency > 10)[0]
        # backtrace the indices to the corresponding parameters
        ALGO2_promotion_efficiency = (ALGO2_misses - LRU_misses) / (
            LRU_promotions - ALGO2_promotions
        )
        if LRU_promotion_efficiency_stack is None:
            LRU_promotion_efficiency_stack = np.hstack(
                (ALGO1_promotion_efficiency, ALGO2_promotion_efficiency)
            )
        else:
            LRU_promotion_efficiency_stack = np.hstack(
                (LRU_promotion_efficiency_stack, ALGO1_promotion_efficiency)
            )
            LRU_promotion_efficiency_stack = np.hstack(
                (LRU_promotion_efficiency_stack, ALGO2_promotion_efficiency)
            )

    ALGOs_params = ALGOs_params
    # LRU_promotion_efficiency_stack = np.hstack((LRU_promotion_efficiency_stack, LRU_promotion_efficiency))

    # adjust the positions
    total_boxes = LRU_promotion_efficiency_stack.shape[1]
    positions = np.arange(1, total_boxes + 1)
    interval = 0.15  # Controls how close the pairs are
    adjusted_positions = []
    for i in range(int((total_boxes) / 2)):  # Group pairs, excluding the last "LRU"
        if i > 2:
            shift = 1
        else:
            shift = 0
        adjusted_positions.append(positions[2 * i] + interval + shift)
        adjusted_positions.append(positions[2 * i + 1] - interval + shift)

    fig, ax = plt.subplots(figsize=(6, 3.8))
    box = ax.boxplot(
        LRU_promotion_efficiency_stack,
        positions=adjusted_positions,
        patch_artist=True,
        showfliers=False,
        whis=[10, 90],
        showmeans=True,
        meanprops=dict(
            marker="o", markerfacecolor="white", markeredgecolor="black", markersize=8
        ),
        medianprops=dict(linestyle="-", linewidth=1.25, color="black"),
    )

    for i, box_values in enumerate(box["boxes"]):
        if i % 2 == 1:
            # use lightpink
            box_values.set(color="black", linewidth=1.25)
            box_values.set(facecolor="lightpink")
            box_values.set_hatch("///")
        else:
            box_values.set(color="black", linewidth=1.25)
            box_values.set(facecolor="lightblue")

    num_params = len(ALGOs_params) * 2
    plt.grid(True, color="lightgray", linestyle="--", linewidth=1)
    plt.xlabel(kwargs.get("xlabel"), fontsize=19)
    # plt.xticks(np.arange(1, num_params + 1), fontsize=18)
    group_centers = np.arange(1, total_boxes // 2 + 1) * 2 - 0.5
    group_centers[-1] = group_centers[-1] + 1
    plt.xticks(group_centers, ALGOs_params, fontsize=18)
    plt.ylabel("Miss reduced per promotion", fontsize=19)
    plt.yticks(fontsize=18)
    plt.savefig(
        kwargs.get("ALGO_name")
        + "_efficiency_boxplot_"
        + kwargs.get("size_cache")
        + ".pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_hot_cache_promotion_insensitivity(
    HOTCache_promotions, Regular_promotions, param_idx, threshold_params, **kwargs
):
    # promotion and miss ratio are of shape (num_files, threshold_params, params)
    # get the average at axis 0

    # plot the promotion insensitivity as a function of the threshold parameter
    print("HOTCache_promotions shape", HOTCache_promotions.shape)
    HOTCache_promotions_param_idx = HOTCache_promotions[:, :, param_idx]
    Regular_promotions_sizematch = np.array(
        [Regular_promotions[:, param_idx] for i in range(len(threshold_params))]
    ).T
    HOTCache_promotions_param_idx = (
        HOTCache_promotions_param_idx / Regular_promotions_sizematch
    )

    # convert the threshold_params to string
    # HOTCache_promotions_param_idx = HOTCache_promotions_param_idx.T
    print("hotcache promotions param idx", HOTCache_promotions_param_idx.shape)

    plt.figure(figsize=(6, 3.8))
    # box = ax.boxplot(ALGO_relative_promotions_stack, positions=adjusted_positions, patch_artist=True, showfliers=False, whis=[10,90], showmeans=True,
    #                   meanprops=dict(marker='o', markerfacecolor='white', markeredgecolor='black', markersize=8),
    #                   medianprops = dict(linestyle='-', linewidth=1.25, color='black'))
    print(np.shape(HOTCache_promotions_param_idx))
    box = plt.boxplot(
        HOTCache_promotions_param_idx,
        patch_artist=True,
        showfliers=False,
        whis=[10, 90],
        showmeans=True,
        meanprops=dict(
            marker="o", markerfacecolor="white", markeredgecolor="black", markersize=8
        ),
        medianprops=dict(linestyle="-", linewidth=1.25, color="black"),
    )
    for i, box_values in enumerate(box["boxes"]):
        box_values.set(color="black", linewidth=1.25)
        box_values.set(facecolor="lightblue")

    plt.xlabel("Threshold parameter", fontsize=19)
    plt.ylabel("Rel. Promotions vs. Original", fontsize=19)
    plt.xticks(
        ticks=range(1, len(threshold_params) + 1), labels=threshold_params, fontsize=18
    )
    plt.yticks(fontsize=18)
    # plt.legend()
    plt.grid(True, color="lightgray", linestyle="--", linewidth=1)
    plt.savefig(
        "Promotion_insensitivity"
        + kwargs.get("size_cache")
        + kwargs.get("ALGO_name")
        + ".pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_hot_cache_miss_insensitivity(
    HOTCache_miss, Regular_miss, param_idx, threshold_params, **kwargs
):
    # promotion and miss ratio are of shape (num_files, threshold_params, params)
    # plot the promotion insensitivity as a function of the threshold parameter
    HOTCache_miss_param_idx = HOTCache_miss[:, :, param_idx]
    Regular_miss_sizematch = np.array(
        [Regular_miss[:, param_idx] for i in range(len(threshold_params))]
    ).T
    HOTCache_miss_param_idx = (
        HOTCache_miss_param_idx / Regular_miss_sizematch
    )  # relative miss ratio

    # convert the threshold_params to string
    print("hotcache promotions param idx", HOTCache_miss_param_idx.shape)

    plt.figure(figsize=(6, 3.8))
    box = plt.boxplot(
        HOTCache_miss_param_idx,
        patch_artist=True,
        showfliers=False,
        whis=[10, 90],
        showmeans=True,
        meanprops=dict(
            marker="o", markerfacecolor="white", markeredgecolor="black", markersize=8
        ),
        medianprops=dict(linestyle="-", linewidth=1.25, color="black"),
    )
    for i, box_values in enumerate(box["boxes"]):
        box_values.set(color="black", linewidth=1.25)
        box_values.set(facecolor="lightblue")

    plt.xlabel("Threshold parameter", fontsize=19)
    plt.ylabel("Rel. Miss Ratio vs. Original", fontsize=19)
    plt.xticks(
        ticks=range(1, len(threshold_params) + 1), labels=threshold_params, fontsize=18
    )
    plt.yticks(fontsize=18)
    plt.grid(True, color="lightgray", linestyle="--", linewidth=1)
    plt.savefig(
        "miss_insensitivity"
        + kwargs.get("size_cache")
        + kwargs.get("ALGO_name")
        + ".pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_belady_clock_miss(ALGO1, ALGO2, FIFO_miss_ratio, j, **kwargs):
    ALGO1 = ALGO1[:, :j, :]
    ALGO2 = ALGO2[:, :j]
    ALGO2 = np.reshape(ALGO2, (ALGO2.shape[0], ALGO2.shape[1], 1))
    FIFO_miss_ratio = FIFO_miss_ratio[:, :j, :]

    # print("algo2 shape", ALGO2.shape)
    # print("fifo shape", FIFO_miss_ratio.shape)

    # interleaving the ALGOi_miss_ratio and ALGO2_miss_ratio
    ALGO_miss_ratio_stack = None
    # print("ALGO1s_miss_ratio shape", ALGO1s_miss_ratio.shape[1])
    for i in range(ALGO1.shape[0]):
        ALGO1_miss_ratio = ALGO1[i]
        ALGO2_miss_ratio = ALGO2[i]
        ALGO1_miss_ratio = ALGO1_miss_ratio / FIFO_miss_ratio[i]
        ALGO2_miss_ratio = ALGO2_miss_ratio / FIFO_miss_ratio[i]
        if ALGO_miss_ratio_stack is None:
            ALGO_miss_ratio_stack = np.hstack((ALGO2_miss_ratio, ALGO1_miss_ratio))
        else:
            ALGO_miss_ratio_stack = np.hstack((ALGO_miss_ratio_stack, ALGO2_miss_ratio))
            ALGO_miss_ratio_stack = np.hstack((ALGO_miss_ratio_stack, ALGO1_miss_ratio))
    fig, ax = plt.subplots(figsize=(6, 3.8))

    # adjust the positions
    # print("stack shape", ALGO_miss_ratio_stack.shape)
    total_boxes = ALGO_miss_ratio_stack.shape[1]
    # print("total boxes", total_boxes)
    positions = np.arange(1, total_boxes + 1)
    shift = 0.15  # Controls how close the pairs are
    adjusted_positions = []
    for i in range(int((total_boxes) / 2)):  # Group pairs, excluding the last "LRU"
        adjusted_positions.append(positions[2 * i] + shift)
        adjusted_positions.append(positions[2 * i + 1] - shift)

    # print("algo miss ratio stack shape", ALGO_miss_ratio_stack.shape)
    box = ax.boxplot(
        ALGO_miss_ratio_stack,
        positions=adjusted_positions,
        patch_artist=True,
        showfliers=False,
        whis=[10, 90],
        showmeans=True,
        meanprops=dict(
            marker="o", markerfacecolor="white", markeredgecolor="black", markersize=8
        ),
        medianprops=dict(linestyle="-", linewidth=1.25, color="black"),
    )
    params = kwargs.get("params")
    len_params = len(params) * 2
    for i, box_values in enumerate(box["boxes"]):
        if i % 2 == 1:
            # use lightpink
            box_values.set(color="black", linewidth=1.25)  # Edge color and line width
            box_values.set(facecolor="lightpink")
            box_values.set_hatch("///")
        else:
            box_values.set(color="black", linewidth=1.25)  # Edge color and line width
            box_values.set(facecolor="lightblue")  # Fill color

    plt.grid(True, color="lightgray", linestyle="--", linewidth=1)
    # plt.xlabel(kwargs.get("xlabel"), fontsize=19)
    plt.xlabel("Cache size", fontsize=19)
    group_centers = np.arange(1, total_boxes // 2 + 1) * 2 - 0.5
    # plt.xticks(np.arange(1, len_params + 1), fontsize=18)
    plt.xticks(group_centers, params, fontsize=18)
    print(params)
    plt.ylabel("Miss ratio relative to FIFO", fontsize=19)
    plt.yticks(fontsize=18)
    plt.savefig(kwargs.get("ALGO_name") + "_boxplot" + ".pdf", bbox_inches="tight")
    plt.close()


def plot_belady_clock_promotion(ALGO1, ALGO2, LRU_promotions, j, **kwargs):
    ALGO1 = ALGO1[:, :j, :]
    ALGO2 = ALGO2[:, :j]
    ALGO2 = np.reshape(ALGO2, (ALGO2.shape[0], ALGO2.shape[1], 1))
    LRU_promotions = LRU_promotions[:, :j, :]

    ALGO_relative_promotions_stack = None
    for i in range(ALGO1.shape[0]):
        ALGO1_promotions = ALGO1[i]
        ALGO2_promotions = ALGO2[i]
        ALGO1_promotions = np.reshape(ALGO1_promotions, (ALGO1_promotions.shape[0], 1))
        ALGO1_promotions = ALGO1_promotions / LRU_promotions[i]
        ALGO2_promotions = np.reshape(ALGO2_promotions, (ALGO2_promotions.shape[0], 1))
        ALGO2_promotions = ALGO2_promotions / LRU_promotions[i]
        if ALGO_relative_promotions_stack is None:
            ALGO_relative_promotions_stack = np.hstack(
                (ALGO2_promotions, ALGO1_promotions)
            )
        else:
            ALGO_relative_promotions_stack = np.hstack(
                (ALGO_relative_promotions_stack, ALGO2_promotions)
            )
            ALGO_relative_promotions_stack = np.hstack(
                (ALGO_relative_promotions_stack, ALGO1_promotions)
            )

    fig, ax = plt.subplots(figsize=(6, 3.8))
    lightpink_legend = mpatches.Patch(
        facecolor="lightpink", edgecolor="black", hatch="///", label="BeladyClock"
    )
    lightblue_legend = mpatches.Patch(
        facecolor="lightblue", edgecolor="black", label="Clock"
    )
    ax.legend(handles=[lightpink_legend, lightblue_legend], fontsize=18)

    total_boxes = ALGO_relative_promotions_stack.shape[1]
    positions = np.arange(1, total_boxes + 1)
    shift = 0.15  # Controls how close the pairs are
    adjusted_positions = []
    for i in range(int((total_boxes) / 2)):  # Group pairs, excluding the last "LRU"
        adjusted_positions.append(positions[2 * i] + shift)
        adjusted_positions.append(positions[2 * i + 1] - shift)

    box = ax.boxplot(
        ALGO_relative_promotions_stack,
        positions=adjusted_positions,
        patch_artist=True,
        showfliers=False,
        whis=[10, 90],
        showmeans=True,
        meanprops=dict(
            marker="o", markerfacecolor="white", markeredgecolor="black", markersize=8
        ),
        medianprops=dict(linestyle="-", linewidth=1.25, color="black"),
    )

    for i, box_values in enumerate(box["boxes"]):
        if i % 2 == 1:
            # use lightpink
            box_values.set(color="black", linewidth=1.25)
            box_values.set(facecolor="lightpink")
            box_values.set_hatch("///")
        else:
            box_values.set(color="black", linewidth=1.25)
            box_values.set(facecolor="lightblue")

    ALGOs_params = kwargs.get("params")
    len_params = len(ALGOs_params) * 2
    plt.grid(True, color="lightgray", linestyle="--", linewidth=1)
    # plt.xlabel(kwargs.get("xlabel"), fontsize=19)
    plt.xlabel("Cache size", fontsize=18)
    group_centers = np.arange(1, total_boxes // 2 + 1) * 2 - 0.5
    # plt.xticks(np.arange(1, len_params + 1), fontsize=18)
    plt.xticks(group_centers, ALGOs_params, fontsize=18)
    plt.ylabel("Promotions relative to LRU", fontsize=19)
    plt.yticks(fontsize=18)
    plt.savefig(
        kwargs.get("ALGO_name") + "_promotion_boxplot" + ".pdf", bbox_inches="tight"
    )
    plt.close()


def plot_belady_clock_promotion_efficiency(
    ALGO1_miss_ratio,
    ALGO2_miss_ratio,
    ALGO1_promotions,
    ALGO2_promotions,
    FIFO_miss_ratio,
    j,
    Request_data_all_all,
    **kwargs,
):
    print("algo1 promotions", ALGO1_promotions)
    ALGO1_miss_ratio = ALGO1_miss_ratio[:, :j, :]
    ALGO2_miss_ratio = ALGO2_miss_ratio[:, :j]
    ALGO2_miss_ratio = np.reshape(
        ALGO2_miss_ratio, (ALGO2_miss_ratio.shape[0], ALGO2_miss_ratio.shape[1], 1)
    )
    ALGO1_promotions = ALGO1_promotions[:, :j, :]
    ALGO2_promotions = ALGO2_promotions[:, :j]
    ALGO2_promotions = np.reshape(
        ALGO2_promotions, (ALGO2_promotions.shape[0], ALGO2_promotions.shape[1], 1)
    )
    FIFO_miss_ratio = FIFO_miss_ratio[:, :j, :]
    Request_data_all_all = Request_data_all_all[:, :j, :]

    print("algo1 misses", ALGO1_miss_ratio)
    print("fifo miss ratio", FIFO_miss_ratio)
    FIFO_misses = FIFO_miss_ratio * Request_data_all_all
    # print("request data is: ", Request_data_all_all)

    ALGO_relative_promotions_stack = None
    for i in range(ALGO1_miss_ratio.shape[0]):
        ALGO1_miss_ratio_single = ALGO1_miss_ratio[i]
        ALGO2_miss_ratio_single = ALGO2_miss_ratio[i]
        ALGO1_promotion_single = ALGO1_promotions[i]
        ALGO2_promotion_single = ALGO2_promotions[i]
        ALGO1_misses = ALGO1_miss_ratio_single * Request_data_all_all[i]
        ALGO2_misses = ALGO2_miss_ratio_single * Request_data_all_all[i]
        # print("algo1_misses", ALGO1_misses)
        # print("fifo_misses - algo1_misses", FIFO_misses[i] - ALGO1_misses)
        ALGO1_promotion_efficiency = (
            FIFO_misses[i] - ALGO1_misses
        ) / ALGO1_promotion_single
        ALGO2_promotion_efficiency = (
            FIFO_misses[i] - ALGO2_misses
        ) / ALGO2_promotion_single
        # print("ALGO2 promotion efficiency", ALGO2_promotion_efficiency)
        if ALGO_relative_promotions_stack is None:
            ALGO_relative_promotions_stack = np.hstack(
                (ALGO2_promotion_efficiency, ALGO1_promotion_efficiency)
            )
        else:
            ALGO_relative_promotions_stack = np.hstack(
                (ALGO_relative_promotions_stack, ALGO2_promotion_efficiency)
            )
            ALGO_relative_promotions_stack = np.hstack(
                (ALGO_relative_promotions_stack, ALGO1_promotion_efficiency)
            )

    fig, ax = plt.subplots(figsize=(6, 4.5))
    total_boxes = ALGO_relative_promotions_stack.shape[1]
    positions = np.arange(1, total_boxes + 1)
    shift = 0.15  # Controls how close the pairs are
    adjusted_positions = []
    for i in range(int((total_boxes) / 2)):  # Group pairs, excluding the last "LRU"
        adjusted_positions.append(positions[2 * i] + shift)
        adjusted_positions.append(positions[2 * i + 1] - shift)
    box = ax.boxplot(
        ALGO_relative_promotions_stack,
        positions=adjusted_positions,
        patch_artist=True,
        showfliers=False,
        whis=[10, 90],
        showmeans=True,
        meanprops=dict(
            marker="o", markerfacecolor="white", markeredgecolor="black", markersize=8
        ),
        medianprops=dict(linestyle="-", linewidth=1.25, color="black"),
    )

    print(
        "np.mean(ALGO_relative_promotions_stack, axis=0): ",
        np.mean(ALGO_relative_promotions_stack, axis=0),
    )

    for i, box_values in enumerate(box["boxes"]):
        if i % 2 == 1:
            # use lightpink
            box_values.set(color="black", linewidth=1.25)
            box_values.set(facecolor="lightpink")
            box_values.set_hatch("///")
        else:
            box_values.set(color="black", linewidth=1.25)
            box_values.set(facecolor="lightblue")
    # lightpink_legend = mpatches.Patch(facecolor='lightpink', edgecolor='black', hatch='///', label='BeladyClock')
    # lightblue_legend = mpatches.Patch(facecolor='lightblue', edgecolor='black', label='Clock')
    # ax.legend(handles=[lightpink_legend, lightblue_legend], fontsize=18)
    ALGOs_params = kwargs.get("params")
    len_params = len(ALGOs_params) * 2
    plt.grid(True, color="lightgray", linestyle="--", linewidth=1)
    # plt.xlabel(kwargs.get("xlabel"), fontsize=19)
    plt.xlabel("Cache size", fontsize=18)
    group_centers = np.arange(1, total_boxes // 2 + 1) * 2 - 0.5
    # plt.xticks(np.arange(1, len_params + 1), fontsize=18)
    plt.xticks(group_centers, ALGOs_params, fontsize=18)
    plt.ylabel("Misses reduced per promotion", fontsize=17.5)
    plt.yticks(fontsize=18)
    plt.savefig("BeladyClock_efficiency_boxplot" + ".pdf", bbox_inches="tight")
    plt.close()


def plot_promotion_efficiency_average(
    ALGOs_promotions,
    LRU_promotions,
    ALGOs_miss_ratio,
    FIFO_miss_ratio,
    LRU_miss_ratio,
    ALGOs_params,
    TRACE_request,
    **kwargs,
):
    LRU_misses = LRU_miss_ratio * TRACE_request
    FIFO_misses = FIFO_miss_ratio * TRACE_request
    print("fifo misses shape", FIFO_misses.shape)

    LRU_promotion_efficiency = (FIFO_misses - LRU_misses) / LRU_promotions

    # expand the FIFO_hit to the same shape as ALGO_promotions
    print("algos promotion shape", ALGOs_promotions.shape)
    TRACE_request = np.tile(TRACE_request, ALGOs_promotions.shape[1])
    print("trace request shape", TRACE_request.shape)
    ALGOs_misses = ALGOs_miss_ratio * TRACE_request
    print("algo_misses shape", ALGOs_misses.shape)

    # traverse the algorithms
    promotion_efficiency_stack = None
    for i in range(ALGOs_promotions.shape[1]):
        ALGO_misses = np.reshape(ALGOs_misses[:, i], (ALGOs_misses.shape[0], 1))
        front = FIFO_misses - ALGO_misses
        back = np.reshape(ALGOs_promotions[:, i], (ALGOs_promotions.shape[0], 1))
        print("back shape", back.shape)
        ALGO_promotion_efficiency = front / back
        if promotion_efficiency_stack is None:
            promotion_efficiency_stack = ALGO_promotion_efficiency
        else:
            promotion_efficiency_stack = np.hstack(
                (promotion_efficiency_stack, ALGO_promotion_efficiency)
            )

    ALGOs_params = ALGOs_params + ["LRU"]

    promotion_efficiency_stack = np.hstack(
        (promotion_efficiency_stack, LRU_promotion_efficiency)
    )

    print("promotion efficiency stack shape", promotion_efficiency_stack.shape)

    fig, ax = plt.subplots(figsize=(6, 3.8))
    box = ax.boxplot(
        promotion_efficiency_stack,
        patch_artist=True,
        showfliers=False,
        whis=[10, 90],
        showmeans=True,
        meanprops=dict(
            marker="o", markerfacecolor="white", markeredgecolor="black", markersize=8
        ),
        medianprops=dict(linestyle="-", linewidth=1.25, color="black"),
    )

    for box_item in box["boxes"]:
        box_item.set(color="black", linewidth=1.25)
        box_item.set(facecolor="lightblue")

    box["boxes"][-1].set_facecolor("lightpink")

    num_params = len(ALGOs_params)
    plt.grid(True, color="lightgray", linestyle="--", linewidth=1)
    plt.xlabel(kwargs.get("xlabel"), fontsize=19)
    plt.xticks(np.arange(1, num_params + 1), ALGOs_params, fontsize=18)
    plt.ylabel("Misses reduced per promotion", fontsize=19)
    plt.yticks(fontsize=18)
    plt.savefig(
        "promotion" + "_efficiency_sum_boxplot_" + kwargs.get("size_cache") + ".pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_promotion_efficiency_average_bar(
    ALGOs_promotions,
    LRU_promotions,
    ALGOs_miss_ratio,
    FIFO_miss_ratio,
    LRU_miss_ratio,
    ALGOs_params,
    TRACE_request,
    **kwargs,
):
    LRU_misses = LRU_miss_ratio * TRACE_request
    FIFO_misses = FIFO_miss_ratio * TRACE_request
    print("fifo misses shape", FIFO_misses.shape)

    LRU_promotion_efficiency = (FIFO_misses - LRU_misses) / LRU_promotions

    # Expand the TRACE_request to match ALGO_promotions
    print("algos promotion shape", ALGOs_promotions.shape)
    TRACE_request = np.tile(TRACE_request, ALGOs_promotions.shape[1])
    print("trace request shape", TRACE_request.shape)
    ALGOs_misses = ALGOs_miss_ratio * TRACE_request
    print("algo_misses shape", ALGOs_misses.shape)

    # Calculate promotion efficiency for all algorithms
    promotion_efficiency_stack = []
    FIFO_misses = FIFO_misses.reshape(FIFO_misses.shape[0])
    for i in range(ALGOs_promotions.shape[1]):
        ALGO_misses = ALGOs_misses[:, i]
        print(np.shape(ALGO_misses), np.shape(FIFO_misses))
        front = FIFO_misses - ALGO_misses
        back = ALGOs_promotions[:, i]
        print(np.shape(front), np.shape(back))
        ALGO_promotion_efficiency = front / back
        print("ALGO promotion efficiency", np.shape(ALGO_promotion_efficiency))
        mean = np.mean(ALGO_promotion_efficiency)
        promotion_efficiency_stack.append(mean)

    LRU_promotion_efficiency = np.mean(LRU_promotion_efficiency)
    # Add LRU efficiency
    promotion_efficiency_stack.append(LRU_promotion_efficiency)
    # print("promotion efficiency stack shape", promotion_efficiency_stack.shape)
    promotion_efficiency_stack = np.array(promotion_efficiency_stack)

    # Bar plot
    fig, ax = plt.subplots(figsize=(6, 3.8))
    bar_width = 0.6
    x_positions = np.arange(len(ALGOs_params) // 2 + 1)
    print(x_positions)
    plt.grid(True, color="lightgray", linestyle="--", linewidth=1, axis="y")

    # bars_1 = ax.bar(x_positions[:-1], promotion_efficiency_stack[:len(ALGOs_params)//2], bar_width, capsize=5, color='blue', edgecolor='black', linewidth=1.25)
    bars_2 = ax.bar(
        x_positions[:-1],
        promotion_efficiency_stack[len(ALGOs_params) // 2 : -1],
        bar_width,
        capsize=5,
        color="lightblue",
        edgecolor="black",
        linewidth=1.25,
    )
    bars_1 = ax.bar(
        x_positions[:-1],
        promotion_efficiency_stack[: len(ALGOs_params) // 2],
        bar_width,
        capsize=5,
        color="steelblue",
        edgecolor="black",
        linewidth=1.25,
    )
    bars_lru = ax.bar(
        x_positions[-1],
        promotion_efficiency_stack[-1],
        bar_width,
        capsize=5,
        color="lightpink",
        edgecolor="black",
        linewidth=1.25,
    )

    # Highlight the LRU bar
    # bars[-1].set_color('lightpink')

    # Add labels and grid
    num_params = len(ALGOs_params)
    ALGOs_params = ALGOs_params[:4] + ["LRU"]
    plt.xlabel("Technique", fontsize=19)
    plt.xticks(x_positions, ALGOs_params, fontsize=18)
    plt.ylabel("Promotion Efficiency", fontsize=19)
    plt.yticks(fontsize=18)

    lightpink_legend = mpatches.Patch(
        facecolor="steelblue", edgecolor="black", label="Original"
    )
    lightblue_legend = mpatches.Patch(
        facecolor="lightblue", edgecolor="black", label="Boosted by HotCache/AGE"
    )
    # ax.legend(handles=[lightpink_legend, lightblue_legend], fontsize=18, ncol=2, loc='lower center')
    plt.legend(
        bbox_to_anchor=(-0.13, 1.1),
        loc="center left",
        handles=[lightpink_legend, lightblue_legend],
        ncol=2,
        fontsize=18,
        frameon=False,
        handletextpad=0.3,
        columnspacing=0.8,
    )

    # Save the figure
    plt.savefig(
        "promotion" + "_efficiency_sum_barplot_" + kwargs.get("size_cache") + ".pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_promotion_offline_hot_cache(hotlru_promotion, ALGO_params, **kwargs):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    box = ax.boxplot(
        hotlru_promotion,
        patch_artist=True,
        showfliers=False,
        whis=[10, 90],
        showmeans=True,
        meanprops=dict(
            marker="o", markerfacecolor="white", markeredgecolor="black", markersize=8
        ),
        medianprops=dict(linestyle="-", linewidth=1.25, color="black"),
    )

    for box_item in box["boxes"]:
        box_item.set(color="black", linewidth=1.25)
        box_item.set(facecolor="lightblue")
    len_params = len(ALGO_params)
    plt.grid(True, color="lightgray", linestyle="--", linewidth=1)
    print("xlabel", kwargs.get("xlabel"))
    plt.xlabel(kwargs.get("xlabel"), fontsize=19)
    plt.xticks(np.arange(1, len_params + 1), ALGO_params, fontsize=18)
    plt.ylabel("Promotions relative to LRU", fontsize=19)
    plt.yticks(fontsize=18)
    # plt.ylim(0, 1)
    plt.savefig(
        "hotlru" + "_promotion_boxplot_" + kwargs.get("size_cache") + ".pdf",
        bbox_inches="tight",
    )
    plt.close()
