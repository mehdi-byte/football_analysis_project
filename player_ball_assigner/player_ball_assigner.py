import sys
sys.path.append('..')
from utils import get_distance

class PlayerBallAssigner():
    def __init__(self):
        self.max_distance = 70

    def assign_ball_to_player(self,player_tracks,ball_box) :
        min_distance = 900
        assigned_player = -1
        ball_center = int((ball_box[0]+ball_box[2])//2),((int(ball_box[1])+int((ball_box[3])))//2)
        #print("ball_position",ball_center)
        for player_id,player in player_tracks.items():
            player_bbox = player["bbox"]
            #print("player position",player_bbox)
            left_position = player_bbox[0],player_bbox[-1]
            left_distance = get_distance(left_position,ball_center)
            right_position = player_bbox[2],player_bbox[-1]
            right_distance = get_distance(right_position,ball_center)
            distance = min(right_distance,left_distance)
            #print(distance)
            if distance < self.max_distance:
                if distance < min_distance:
                    min_distance = distance
                    assigned_player = player_id

        return assigned_player