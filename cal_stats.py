import sys
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    file_name = sys.argv[-1]
    df = pd.read_csv(file_name, index_col=0)
    print(df, end='\n\n')

    plt.figure(figsize=(25, 6), dpi=80)
    unique_frames_df = df.drop_duplicates("world_index")
    total_unique_frames = unique_frames_df['world_index'].nunique()
    plt.scatter(unique_frames_df["world_index"], unique_frames_df['is_baby'])
    # plt.show()

    detected_baby_df = df[df['is_baby'] == True]
    total__detected_frames = detected_baby_df['world_index'].nunique()
    print(f"Total frames having baby {total__detected_frames}, i.e {total__detected_frames/total_unique_frames*100:.2f}%")

    gaze_in_box_df = detected_baby_df[detected_baby_df['in_bounding_box']==True]
    gaze_in_box = gaze_in_box_df['world_index'].nunique()
    print(f"Loooking at baby (in bounding box): {gaze_in_box}, i.e {gaze_in_box/total__detected_frames*100:.2f}%")

    gaze_in_seg_df = detected_baby_df[detected_baby_df['in_segmentation']==True]
    gaze_in_seg = gaze_in_seg_df['world_index'].nunique()
    print(f"Loooking at baby (in segmentation): {gaze_in_seg}, i.e {gaze_in_seg/total__detected_frames*100:.2f}%")



    