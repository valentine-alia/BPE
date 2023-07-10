import numpy as np
import random
from scipy.spatial import Delaunay

def samples_above_threshold(vp, true_value, n_samples, lower_thresh_const):
    hit  = False # to check if there are more than one true value
    try:
        true_value[0][0] 
    except:
        hit = True

    if hit:
        threshold = vp.pdf(true_value)
    else:
        threshold = min(vp.pdf(np.array(true_value))) # grabs the smallest pdf if there are more than one true value
        
    threshold_edited = threshold * lower_thresh_const
    Xs, _ = vp.sample(n_samples * 5) # get samples out of the distribution, do 5 times to make sure you hit the n_samples below
    pdfs = vp.pdf(Xs) # find the pdf of each of those samples  

    high_threshold_samples = []

    for i in range(len(pdfs)): # the while + for loop wasn't working so I just had it add samples up to n_samples and check when to break
        if pdfs[i] >= threshold_edited and len(high_threshold_samples) <= n_samples: 
            high_threshold_samples.append(Xs[i])
            if len(high_threshold_samples) == n_samples:
                break

    return np.array(high_threshold_samples)

def minkowskiDist(v1, v2):
    #Assumes v1 and v2 are equal length arrays of numbers
    dist = 0
    for i in range(len(v1)):
        dist += abs(v1[i] - v2[i])**2
    return dist**(1/2)

class Example(object):
    
    def __init__(self, features):
        #Assumes features is an array of floats
        self.features = features

    def dimensionality(self):
        return len(self.features)

    def getFeatures(self):
        return self.features[:]
    
    def distance(self, other):
        return minkowskiDist(self.features, other.getFeatures())
    
class Cluster(object):
    
    def __init__(self, examples):
        """Assumes examples a non-empty list of Examples"""
        self.examples = examples
        self.centroid = self.computeCentroid()
        
    def update(self, examples):
        """Assume examples is a non-empty list of Examples
           Replace examples; return amount centroid has changed"""
        oldCentroid = self.centroid
        self.examples = examples
        self.centroid = self.computeCentroid()
        return oldCentroid.distance(self.centroid)
    
    def computeCentroid(self):
        vals = np.array([0.0]*self.examples[0].dimensionality())
        for e in self.examples: #compute mean
            vals += e.getFeatures()
        centroid = Example(vals/ len(self.examples))
        return centroid

    def getCentroid(self):
        return self.centroid

    def inertia(self):
        totDist = 0
        for e in self.examples:
            totDist += (e.distance(self.centroid))**2
        return totDist
        
    def members(self):
        for e in self.examples:
            yield e  

def kmeans(examples, k, verbose = False):
    #Get k randomly chosen initial centroids, create cluster for each
    initialCentroids = random.sample(examples, k)
    clusters = []
    for e in initialCentroids:
        clusters.append(Cluster([e]))
        
    #Iterate until centroids do not change
    converged = False
    numIterations = 0
    while not converged:
        numIterations += 1
        #Create a list containing k distinct empty lists
        newClusters = []
        for i in range(k):
            newClusters.append([])
            
        #Associate each example with closest centroid
        for e in examples:
            #Find the centroid closest to e
            smallestDistance = e.distance(clusters[0].getCentroid())
            index = 0
            for i in range(1, k):
                distance = e.distance(clusters[i].getCentroid())
                if distance < smallestDistance:
                    smallestDistance = distance
                    index = i
            #Add e to the list of examples for appropriate cluster
            newClusters[index].append(e)

        for c in newClusters: #Avoid having empty clusters
            if len(c) == 0:
                raise ValueError('Empty Cluster')
        
        #Update each cluster; check if a centroid has changed
        converged = True
        for i in range(k):
            if clusters[i].update(newClusters[i]) > 0.0:
                converged = False
        #plotClusters(clusters)
        if verbose:
            print('Iteration #' + str(numIterations))
            for c in clusters:
                print(c)
            print('') #add blank line

    return clusters

def totalInertia(clusters):
    """Assumes clusters a list of clusters
       Returns a measure of the total dissimilarity of the
       clusters in the list"""
    totDist = 0
    for c in clusters:
        totDist += c.inertia()
    return totDist

def trykmeans(examples, numClusters, numTrials, verbose = False):
    """Calls kmeans numTrials times and returns the result with the
          lowest dissimilarity"""
    best = kmeans(examples, numClusters, verbose)
    trial = 1
    while trial < numTrials:
        try:
            clusters = kmeans(examples, numClusters, verbose)
        except ValueError:
            continue #If failed, try again
        if totalInertia(clusters) < totalInertia(best):
            best = clusters
        trial += 1
    return best

def getDataFeatures(pts): #n-dimensional version
    points = []
    dim = len(pts[0])

    for i in range(len(pts)):
        point = []
        for j in range(dim):
            point.append(pts[i][j])
        points.append(Example(np.array(point)))
        
    return points

def get_cluster_points(try_kmeans):
    try_k = try_kmeans
    all_clusters = []
    for cluster in try_k:
        cluster_pts = []
        examples_cluster = cluster.examples
        for examp in examples_cluster:
            cluster_pts.append(examp.getFeatures())
        all_clusters.append(np.array(cluster_pts))
    return all_clusters

def get_Delaunay(vp, clusters): # this works for 2-D NOT USED ANYMORE

    integrals = 0
    for i in range(len(clusters)):
        tri = Delaunay(clusters[i])
        indices = tri.simplices
        vertices = clusters[i][indices]
        areas = []
        centroids = []

        for pnt in vertices:
            Ax, Ay, Bx, By, Cx, Cy = pnt[0][0], pnt[0][1], pnt[1][0], pnt[1][1], pnt[2][0], pnt[2][1]
            areas.append(abs(Ax*(By-Cy) + Bx*(Cy-Ay) + Cx*(Ay-By)))
            centroids.append([(Ax+Bx+Cx)/2, (Ay+By+Cy)/2])

        pdfs = vp.pdf(centroids)
        integral = np.array(areas) @ pdfs
        integrals += integral
    
    return integrals
    

def deter(pnts):
    v0 = np.array(pnts[0])
    n = len(pnts) - 1
    arry = []
    centroid = []

    for i in range(n + 1):
        if i != n:
            centroid.append(sum(pnts[:,i]) / (n+1)) # just gets all the x values and averages them, then y values, etc
        if i == 0:
            continue
        arry.append(np.array(pnts[i]) - v0) # making an matrix of vectors from one vertex to another (all to v0)

    arry_ = np.array(arry)
    det = abs(np.linalg.det(arry_))
    vol = det/np.math.factorial(n)

    return vol, centroid

def get_Delaunay_integral_n(vp, clusters, num_clusts):

    integrals = 0
    if num_clusts == 0:
        tri = Delaunay(clusters)
        indices = tri.simplices
        vertices = clusters[indices]
        
        vols = []
        centroids = []
        for pnts in vertices:
            det = deter(pnts)
            vols.append(det[0])
            centroids.append(det[1])
        
        pdfs = vp.pdf(np.array(centroids)) # pdf of the centroids
        integrals = np.array(vols) @ pdfs # just the dot product

    else:
        for i in range(len(clusters)):
            tri = Delaunay(clusters[i])
            indices = tri.simplices
            vertices = clusters[i][indices]
            
            vols = []
            centroids = []
            for pnts in vertices:
                det = deter(pnts)
                vols.append(det[0])
                centroids.append(det[1])
            
            pdfs = vp.pdf(np.array(centroids))
            integral = np.array(vols) @ pdfs

            integrals += integral
    
    return integrals

def Del_integral_n_dim(vp, true_value, n_samples, num_clusts, lower_thresh_const):
    samp = samples_above_threshold(vp, true_value, n_samples, lower_thresh_const)
    if num_clusts == 0:
        integral = get_Delaunay_integral_n(vp, samp, num_clusts)
    else:
        make_data_features = getDataFeatures(samp)
        get_k_means = trykmeans(make_data_features, num_clusts, 5)
        clust = get_cluster_points(get_k_means)
        integral = get_Delaunay_integral_n(vp, clust, num_clusts)
    return integral