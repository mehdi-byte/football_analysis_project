from utils import read_video, write_video
from trackers import Tracker
import supervision as sv
from assign import Assigner
import cv2
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import Cameramovementestimator
from view_transformer import  View_transformer
from distance_and_speed_estimator import DistanceSpeed_Estimator
def main():
    #print(sv.__version__)
    video_frames = read_video("data_videos/08fd33_4.mp4")
    tracker = Tracker("models/best100.pt")
    tracks = tracker.get_objects_tracks(video_frames,save_path="final_tracks/tracks.json",read_path=True)
    tracker.add_positions_to_track(tracks)
    # camera movement 
    camera_movement_estimator = Cameramovementestimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,read_from_stub=True,stub_path="final_tracks/camera_movement.pkl")
    camera_movement_estimator.add_adjust_positions(tracks,camera_movement_per_frame)

    #view_transformer 
    view_transformer_object = View_transformer()
    view_transformer_object.add_position_to_tracks(tracks)
    #print("camera movement ", camera_movement_per_frame)
    #interpolate ball positions
    #print(tracks["ball"])
    tracks["ball"] = tracker.interpolate_football_positions(tracks["ball"])
    # Speed and distance estimator
    speed_distance_estimator = DistanceSpeed_Estimator()
    speed_distance_estimator.add_speed_and_distance_to_tracks(tracks)
    team_assigner = Assigner()
    team_assigner.assign_team_color(video_frames[0],tracks["player"][0])
    for frame_num,players in enumerate(tracks["player"]):
        for player_id,track in players.items():
            team =team_assigner.get_player_team(video_frames[frame_num],track["bbox"],player_id)
            tracks["player"][frame_num][player_id]["team"] = team
            tracks["player"][frame_num][player_id]["color"] = team_assigner.teams_colors[team]

    #assign ball to player
    player_ball_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num,player_track in enumerate(tracks["player"]):
        ball_position = tracks["ball"][frame_num]["1"]["bbox"]
        #print(ball_position)
        player_id = player_ball_assigner.assign_ball_to_player(player_track,ball_position)
        if player_id != -1:
            tracks["player"][frame_num][player_id]["has_ball"] = True
            team_ball_control.append(tracks["player"][frame_num][player_id]["team"])
        else:
            team_ball_control.append(team_ball_control[-1])
    #draw_object
    print("coordinates player" ,tracks["player"][677]["66"])
    print("coordinates ball",tracks["ball"][677]["1"])
    #draw elipses
    output_frames_video = tracker.draw_annotations(video_frames,tracks,team_ball_control)
    #draw speed and distance
    output_frames_video = speed_distance_estimator.draw_speed_and_distance(output_frames_video,tracks)
    #draw camera movement
    output_frames_video = camera_movement_estimator.draw_camera_movement(output_frames_video,camera_movement_per_frame)
    write_video(output_frames_video,"outputs/output_video.mp4")


if __name__ == "__main__":
    main()