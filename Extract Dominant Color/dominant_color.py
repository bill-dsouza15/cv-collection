# Extract Dominant Color
#
# The module contains the DominantColor class which 
# provides the extractColor function for obtaining the list
# of dominant colors


import cv2
from sklearn.cluster import KMeans

class DominantColor():
    clusters = None

    def __init__(self, image, clusters=3):
        self.image = image
        self.clusters = clusters

    def extractColor(self):
        # Convert to RGB
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)


        # Reshape image
        h, w, c = image.shape
        image = image.reshape((h*w, c))


        # Using K-Means Clustering
        kmeans = KMeans(self.clusters)
        kmeans.fit(image)


        # Cluster centres = dominant colors
        colors = kmeans.cluster_centers_


        return colors.astype(int)
