import numpy as np
import cv2

class View_transformer():
    def __init__(self):
        width_court = 105
        height_court = 23.32

        self.pixel_vertices = np.array([
            [110, 1035],  # Top-left
            [265, 275],  # Top-right
            [910, 260],  # Bottom-right
            [1640, 915]  # Bottom-left
        ], np.float32)
        self.target_vertices = np.array([
            [0, width_court],
            [0, 0],
            [0, height_court],
            [width_court, height_court]
        ], np.float32)

        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)
    def transform_point(self,point):
        p = (int(point[0]), int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >=0
        if not is_inside:
            return None
        reshaped_point = point.reshape(-1,1,2).astype(np.float32)
        transformed_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)
        
        return transformed_point.reshape(-1,2)
    def add_position_to_tracks(self,tracks):
        for object,object_tracks in tracks.items():
            for num_frame,track in enumerate(object_tracks):
                for track_id,track_info in track.items():
                    position = track_info["position_adjusted"]
                    position = np.array(position)
                    postion_transformed = self.transform_point(position)
                    tracks[object][num_frame][track_id]["position_transformed"] = postion_transformed
        return tracks