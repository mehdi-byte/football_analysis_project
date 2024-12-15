import sys
sys.path.append('../')
from utils import get_distance
import cv2

class DistanceSpeed_Estimator():
    def __init__(self) :
        self.window_frame = 5
        self.frame_rate = 25
    def add_speed_and_distance_to_tracks(self,tracks):
        total_distance_covered = {}
        for object,object_tracks in tracks.items():
            if object == "ball" or object == "referee":
                continue
            num_frames = len(object_tracks)
            for num_frame in range(1,num_frames,self.window_frame):
                last_frame = min(num_frame+self.window_frame,num_frames-1)
                for track_id ,_ in object_tracks[num_frame].items():
                    if track_id not in object_tracks[last_frame]:
                        continue
                    strat_position = object_tracks[num_frame][track_id]["position_adjusted"]
                    end_position = object_tracks[last_frame][track_id]["position_adjusted"]
                    if strat_position is None or end_position is None:
                        continue
                    distance_covered = get_distance(strat_position,end_position)
                    
                    time = (last_frame-num_frame)/self.frame_rate
                    speed_covered = distance_covered/time
                    speed_covered_kmh = speed_covered*0.36
                    print(f"start position: {strat_position}")
                    print(f"end position: {end_position}")
                    print(f"frame: {last_frame}")
                    print(f"Distance: {distance_covered:.2f} m")
                    print(f"Speed: {speed_covered_kmh:.2f} km/h")
                    print(f"time: {time:.2f} s")
                    if object not in total_distance_covered:
                        total_distance_covered[object] = {}
                    if track_id not in total_distance_covered[object]:
                        total_distance_covered[object][track_id] = 0
                    total_distance_covered[object][track_id] += distance_covered
                    for frame_num_batch in range(num_frame,last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        tracks[object][frame_num_batch][track_id]["speed"] = speed_covered_kmh
                        tracks[object][frame_num_batch][track_id]["distance"] = total_distance_covered[object][track_id]

    def draw_speed_and_distance(self,videos_frames_output,tracks):
        output_frames = []
        for frame_num,frame in enumerate(videos_frames_output):
            for object,object_tracks in tracks.items():
                if object == "ball" or object == "referee":
                    continue
                for track_id,track_info in object_tracks[frame_num].items():
                    if "speed" in track_info:
                        speed = track_info.get("speed",None)
                        distance = track_info.get("distance",None)
                        if speed is None and distance is  None:
                            continue
                        bbox = track_info["bbox"]
                        position = (int((bbox[0]+bbox[2])/2),int(bbox[3]))
                        position = list(position)
                        position = tuple(map(int,position))
                        cv2.putText(frame,f"Speed: {speed:.2f} km/h",position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
                        cv2.putText(frame,f"Distance: {distance:.2f} m",(position[0],position[1]+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
            
            output_frames.append(frame)
        return output_frames


                    



