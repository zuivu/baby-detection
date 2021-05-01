import sys
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    file_name = sys.argv[-1]
    df = pd.read_csv(file_name, index_col=0)
    print(df)

    unique_frame = df.drop_duplicates("world_index")
    plt.figure(figsize=(25, 6), dpi=80)
    plt.scatter(unique_frame["world_index"], unique_frame['is_baby'])
    detected_baby_df = df[df['is_baby'] == True]
    print("Total frames having baby", detected_baby_df['world_index'].nunique())

    gaze_in_box = detected_baby_df[detected_baby_df['in_bounding_box']==True]
    print("Loooking at box of baby", gaze_in_box['world_index'].nunique())

    gaze_in_seg = detected_baby_df[detected_baby_df['in_segmentation']==True]
    print("Loooking at segmentation of baby", gaze_in_seg['world_index'].nunique())