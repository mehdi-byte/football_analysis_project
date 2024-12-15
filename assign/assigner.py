import cv2
from sklearn.cluster import KMeans
class Assigner():
    def __init__(self):
        self.teams_colors = {}
        
        self.player_team_dict = {}
        pass
    def get_clustering_model(self,image):
        kmeans = KMeans(n_clusters=2,init="k-means++",n_init=1)
        kmeans=kmeans.fit(image)
        return kmeans

    def get_color(self,frame,bbox):
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
        top_half = image[:image.shape[0]//2,:]
        top_half_2d = top_half.reshape(-1,3)
        kmeans = self.get_clustering_model(top_half_2d)
        labels = kmeans.labels_
        clustered_image = labels.reshape(top_half.shape[0],top_half.shape[1])
        corner_clusters = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters),key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster
        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color

    def assign_team_color(self, frame,players_detections):
        player_colors = []
        for _,player in players_detections.items():
            bbox = player["bbox"]
            player = self.get_color(frame,bbox)
            player_colors.append(player)
        kmeans = KMeans(n_clusters=2,init = "k-means++",n_init=1)
        kmeans = kmeans.fit(player_colors)
        self.teams_colors[1] = kmeans.cluster_centers_[0]
        self.teams_colors[2] = kmeans.cluster_centers_[1]
        self.kmeans = kmeans

    def get_player_team(self,frame,bbox,player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        player_color = self.get_color(frame,bbox)
        team = (self.kmeans).predict(player_color.reshape(1,-1))[0]
        team+=1
        self.player_team_dict[player_id] = team
        return team