import os
import sys
import cv2
import time
import pandas as pd

if __name__  == "__main__":

    # Get files
    date_recording, session_number = sys.argv[1:]
    recording_dir = os.path.join(date_recording, session_number)
    gaze_data_dir = os.path.join(recording_dir, 'exports', os.listdir(os.path.join(recording_dir, 'exports'))[0])

    gaze_df = pd.read_csv(os.path.join(gaze_data_dir, 'gaze_positions.csv'))
    
    # Get gaze data
    world_vid = os.path.join(recording_dir, 'world.mp4')
    cap = cv2.VideoCapture(world_vid)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))       # 1280 px
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))     # 720 px
    print("frame rate:", frame_rate, "width:", width, "height:", height)

    # Export
    os.makedirs(os.path.join("output", date_recording), exist_ok=True)
    video_out = cv2.VideoWriter(filename=os.path.join("output", date_recording, f"{session_number}_world_view_with_detection.avi"), 
                                apiPreference=cv2.CAP_ANY,
                                fourcc=cv2.VideoWriter_fourcc(*"XVID"), 
                                fps=frame_rate, 
                                frameSize=(width, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for f in range(frame_count):
        ret, frame = cap.read()
        if ret:
            gaze_datum = gaze_df[gaze_df['world_index']==f]
            x_norm_coor, y_norm_coor = gaze_datum['norm_pos_x'].to_numpy(), gaze_datum['norm_pos_y'].to_numpy()
            #print(x_norm_coor, y_norm_coor) 

            x_img_coor, y_img_coor = (x_norm_coor*width).astype(int), ((1-y_norm_coor)*height).astype(int)
            for sub_frame in range(len(gaze_datum)):
                cv2.circle(frame, 
                           center=(x_img_coor[sub_frame], y_img_coor[sub_frame]), 
                           radius=10, 
                           color=(0, 0, 255), 
                           thickness=-1)
                #print(x_img_coor[sub_frame], y_img_coor[sub_frame])
            time.sleep(0.01) # Add some delay to avoid processing too fast

            cv2.imshow('Recording', frame)
            video_out.write(frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                sys.exit()

    cap.release()
    video_out.release() 

# EOF