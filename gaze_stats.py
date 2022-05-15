import sys
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    file_name = sys.argv[-1]
    df = pd.read_csv(file_name, index_col=0)
    print(df, end="\n\n")

    # Stats
    detected_baby_df = df[df["is_baby"] == True]
    total_detected_frames = detected_baby_df["world_index"].nunique()
    total_unique_frames = df["world_index"].nunique()
    print(
        f"Total frames having baby {total_detected_frames}, \
        i.e {total_detected_frames/total_unique_frames*100:.2f}% of total frames"
    )

    gaze_in_box_df = detected_baby_df[detected_baby_df["in_bounding_box"] == True]
    gaze_in_box = gaze_in_box_df["world_index"].nunique()
    print(
        f"Loooking at baby (in bounding box): {gaze_in_box}, \
            i.e {gaze_in_box/total_detected_frames*100:.2f}% of frames having baby"
    )

    gaze_in_seg_df = detected_baby_df[detected_baby_df["in_segmentation"] == True]
    gaze_in_seg = gaze_in_seg_df["world_index"].nunique()
    print(
        f"Loooking at baby (in segmentation): {gaze_in_seg}, \
        i.e {gaze_in_seg/total_detected_frames*100:.2f}% of frames having baby"
    )

    # Plot
    df = df.drop_duplicates("world_index")
    yes_baby = df[df["is_baby"] == True]
    no_baby = df[df["is_baby"] == False]
    gaze_in_box_df = gaze_in_box_df.drop_duplicates(subset=["world_index"])
    gaze_in_seg_df = gaze_in_seg_df.drop_duplicates(subset=["world_index"])

    plt.figure(figsize=(10, 3), dpi=120)
    plt.scatter(yes_baby["world_index"], yes_baby["is_baby"], marker=".", label="yes baby")
    plt.scatter(no_baby["world_index"], no_baby["is_baby"], marker=".", label="no baby")
    plt.scatter(
        gaze_in_box_df["world_index"],
        gaze_in_box_df["in_bounding_box"],
        c="r",
        marker=2,
        alpha=0.4,
        label="gaze in box",
    )
    plt.scatter(
        gaze_in_seg_df["world_index"],
        gaze_in_seg_df["in_segmentation"],
        c="g",
        marker=3,
        alpha=0.5,
        label="gaze segmentation",
    )
    plt.yticks([0, 1], labels=["Not looking at baby", "Looking at baby"])
    plt.legend(loc="best")
    plt.grid()
    plt.tight_layout()
    plt.show()
