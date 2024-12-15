from ultralytics import YOLO
import supervision as sv
import json
import cv2
import os
import numpy as np
import pandas as pd

class Tracker():
    def __init__(self,model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
    def  add_positions_to_track(self,tracks):
        for object,object_tracks in tracks.items():
            for num_frame,object_frame in enumerate(object_tracks):
                for track_id,object_track in object_frame.items():
                    x1,y1,x2,y2 = object_track["bbox"]
                    if object == "ball":
                        position = int((x1+x2)/2),int((y1+y2)/2)
                    else:
                        position = int((x1+x2)/2),y2
                    tracks[object][num_frame][track_id]["position"] = position
        
    def interpolate_football_positions(self,ball_positions):
        ball_positions = [x.get('1',{}).get("bbox",[]) for x in ball_positions]
        #print(ball_positions)
        df_ball_positions = pd.DataFrame(ball_positions,columns=["x1","y1","x2","y2"])
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        ball_positions = [{'1':{"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions
    def detect_frames(self,frames):
        detections = []
        batch_size = 20
        for i in range(0,len(frames),batch_size):
            batch_frames = frames[i:i+batch_size]
            batch_detections = self.model.predict(batch_frames)
            detections.extend(batch_detections)
        return detections

    def get_objects_tracks(self,frames,save_path=None,read_path=False):
        tracks = {
            "player": [],
            "ball": [],
            "referee": [],
        }
        if read_path and save_path is not None and os.path.exists(save_path):
            with open(save_path,"r") as f:
                tracks = json.load(f)
            return tracks
        detections = self.detect_frames(frames)
        for num_frame,detection in enumerate(detections):
            cls_names = detection.names
            cls_inv = {cls:k for k,cls in cls_names.items()}
            detection_supervison = sv.Detections.from_ultralytics(detection)
            for obj_ind , class_id in enumerate(detection_supervison.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervison.class_id[obj_ind] = cls_inv["player"]
            detection_tracks = self.tracker.update_with_detections(detection_supervison)
            tracks["player"].append({})
            tracks["ball"].append({})
            tracks["referee"].append({})
            for detection_frame in detection_tracks:
                bbox = detection_frame[0].tolist()
                clss_id = detection_frame[3]
                track_id = detection_frame[4]
                if clss_id == cls_inv["player"]:
                    tracks["player"][num_frame][int(track_id)] = {"bbox":bbox}
                elif clss_id == cls_inv["referee"]:
                    tracks["referee"][num_frame][int(track_id)] = {"bbox":bbox}
            for detection in detection_supervison:
                bbox = detection[0].tolist()
                clss_id = detection[3]
                track_id = detection
                if clss_id == cls_inv["ball"]:
                    tracks["ball"][num_frame]['1'] = {"bbox":bbox}
        print(tracks)
        if save_path is not None:
            with open(save_path,"w") as f:
                json.dump(tracks,f)
        return tracks   
    def draw_elipse(self,frame,bbox,color,track_id):
        x1,y1,x2,y2 = bbox
        x_center,y_center = int((x1+x2)/2),int(y2)
        bbox_width = int(x2) - int(x1)
        cv2.ellipse(frame,center = (x_center,y_center),axes =(int(bbox_width),int(0.35*bbox_width)),angle=0,startAngle=45,endAngle=235,color=color,thickness=2,lineType=cv2.LINE_AA)
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (int(y2) - rectangle_height//2) + 10
        y2_rect = (int(y2) + rectangle_height) +   10
        if track_id is not None:
            cv2.rectangle(frame,(x1_rect,y1_rect),(x2_rect,y2_rect),color,cv2.FILLED)
            cv2.putText(frame,str(track_id),(x1_rect+5,y1_rect+15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
        return frame

    def draw_triangle(self, frame, bbox, color):
        x1, y1, x2, y2 = bbox
        x_center, y_center = int((x1 + x2) / 2), int((y1 + y2) / 2)
        
        # Define the vertices of the triangle
        vertices = np.array([
            [x_center, y_center],  # Top vertex
            [x_center-10, y_center-20],        # Bottom-left vertex
            [x_center+10,  y_center-20]         # Bottom-right vertex
        ], np.int32)
        
        # Reshape the vertices array for polylines/fillPoly
        vertices = vertices.reshape((-1, 1, 2))
        
        # Draw the triangle (filled)
        cv2.fillPoly(frame, [vertices], color)
    
        return frame
    def draw_ball_control(self,frame,num_frame,team_ball_control):
        overlay = frame.copy()
        cv2.rectangle(overlay,(1350,850),(1900,970),(255,255,255),cv2.FILLED)
        alpha = 0.4
        cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)
        #calculate the percentage ball control
        team_ball_control_till_frame = team_ball_control[:num_frame+1]
        team_1_ball_control = team_ball_control_till_frame.count(1)
        team_2_ball_control = team_ball_control_till_frame.count(2)
        team_1 = team_1_ball_control/(team_1_ball_control+team_2_ball_control)
        team_2 = team_2_ball_control/(team_1_ball_control+team_2_ball_control)
        cv2.putText(frame,f"Team 1 : {team_1*100:.2f}%",(1400,900),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
        cv2.putText(frame,f"Team 2: {team_2*100:.2f}%",(1400,950),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
        return frame
    
    def draw_annotations(self,frames,tracks,team_ball_control):
        output_videos_frame = []
        for num_frame,frame in enumerate(frames):
            player_dict = tracks["player"][num_frame]
            ball_dict = tracks["ball"][num_frame]
            referee_dict = tracks["referee"][num_frame]
            for track_id,player in player_dict.items():
                color = player.get("color",(0,255,0))
                frame=self.draw_elipse(frame,player["bbox"],color,track_id)
                
                if player.get("has_ball",False):
                    frame = self.draw_triangle(frame,player["bbox"],(0,0,0))
                    print(num_frame,track_id)
            for track_id,referee in referee_dict.items():
                frame=self.draw_elipse(frame,referee["bbox"],(255,0,0),track_id)
            if len(ball_dict)!=0:
                frame = self.draw_triangle(frame,ball_dict['1']["bbox"],(0,0,255))
            #Draw team ball control
            frame = self.draw_ball_control(frame,num_frame,team_ball_control)
            output_videos_frame.append(frame)
        return output_videos_frame
    
        
          
