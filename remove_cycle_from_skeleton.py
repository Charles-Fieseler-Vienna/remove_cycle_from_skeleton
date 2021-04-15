import tifffile
import scipy
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize
from scipy import sparse
# import argh



def _pixel_neighbours(img, p):
    """Returns the indices of the pixel neighbors"""
    rows, cols = img.shape

    i, j = p[0], p[1]

    rmin = i - 1 if i - 1 >= 0 else 0
    rmax = i + 1 if i + 1 < rows else i

    cmin = j - 1 if j - 1 >= 0 else 0
    cmax = j + 1 if j + 1 < cols else j

    neighbours = []
    for x in range(rmin, rmax + 1):
        for y in range(cmin, cmax + 1):
            neighbours.append([x, y])
    neighbours.remove([p[0], p[1]])

    return neighbours


def _img_to_graph(img):
    """
    Converts an image to a graph, with an edge if both pixels are nonzero
    """
    
    n_x, n_y = img.shape
    n_vertices = n_x*n_y
    edges = []
    
    for i in range(n_x):
        for j in range(n_y):
            p = (i,j)
            if not img[p[0],p[1]] > 0:
                continue
            neighbors = _pixel_neighbours(img, p)
            p_flat = np.ravel_multi_index(p, (n_x, n_y))
            for n in neighbors:
                weight = img[n[0],n[1]]
                if weight > 0:
                    n_flat = np.ravel_multi_index(n, (n_x, n_y))
                    edges.append([p_flat, n_flat, weight])
    return edges
    

def _find_closest_node_to_coordinates(G, coord, sz):
    """Finds closest graph node to pixel coordinates"""
    
    node = None
    dist = np.Inf
    for n in G.nodes():
        this_ij = np.unravel_index(n, sz)
        tmp_dist = np.linalg.norm(coord - this_ij)
        if tmp_dist < dist:
            node = n
            dist = tmp_dist
            
    return node, dist
            
            
def find_intersection_nodes(G, verbose=0):
    # Get intersection node (has degree >2)
    intersections = np.where(np.array(G.degree)[:,1] > 2)[0]
    if len(intersections) > 1:
        print("Found multiple intersection points... this algorithm may not work")
        print("Trying with only the first intersection node")
    intersections = intersections[0]
    # Convert back to Graph index
    intersections = list(G.nodes())[intersections]
    if verbose >= 1:
        print(f"Found intersection node: {intersections}")
    return intersections
    
    
def test_removing_neighbors(G, head_node, tail_node, intersections, verbose=0):
    """
    Check shortest paths when neighbors of the intersection are removed
    Assume the real path should be nearly the length of the entire graph
    """
    target_len = len(G)
    raw_path = nx.shortest_path(G, tail_node, head_node)
    best_path_len = len(raw_path)
    if verbose >= 1:
        print(f"Original path length is {best_path_len}, with target length of {target_len}")

    best_deletion = None
    for n in G.neighbors(intersections):
        G_tmp = G.copy()
        G_tmp.remove_node(n)
        try:
            path = nx.shortest_path(G_tmp, tail_node, head_node)
        except:
            continue
        path_len = len(path)
        if abs(path_len-target_len) < abs(best_path_len-target_len):
            best_path_len = path_len
            best_deletion = n
    if verbose >= 1:
        print(f"Deletion of node {best_deletion} gives a path of length {best_path_len}")
    
    return best_deletion
    
    
def skeletonize_and_remove_cycle(img, head_ij, tail_ij, to_save=False):
    """
    Skeletonize a binary image and enforce a single path from head to tail
        This path should traverse most of the graph
    
    Algorithm:
    1. Skeletonize the image
    2. Convert the image to a graph, with an edge if both pixels are >0
    3. Find head and tail nodes from original pixel annotation (choose the closest skeleton pixel)
    4. Find the intersection (node with degree >2)
    5. Remove neighbors of the intersection node, and find the shortest path
    6. Choose the longest shortest path, which should traverse the entire graph
    7. Return the image with that graph node deleted
    
    Parameters:
    ======================
    img - np.array()
        The 2d binarized image
    head_ij - 2-element tuple
        Pixel annotation of the head; not necessary to be on a skeletonized pixel
    tail_ij - 2-element tuple
        Pixel annotation of the head
    
    Returns:
    =======================
    img_new - np.array()
        Skeletonized image with one pixel removed, so that there is only one shortest path
    """
    if type(img)==str:
        img = tifffile.imread(img)
    # Skeletonize
    img_sk = skeletonize(np.array(img,dtype=bool))
    # Create graph
    edges = _img_to_graph(img_sk)
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    # Find special nodes: head, tail, and intersection
    sz = img.shape
    head_node, _ = _find_closest_node_to_coordinates(G, head_ij, sz)
    tail_node, _ = _find_closest_node_to_coordinates(G, tail_ij, sz)
    intersections = find_intersection_nodes(G)
    # Remove cycle and produce final image
    best_deletion = test_removing_neighbors(G, head_node, tail_node, intersections)
    img_new = img_sk.copy()
    deletion_ij = np.unravel_index(best_deletion, sz)
    img_new[deletion_ij] = False
    
    if to_save:
        fname = 'skeleton_without_cycle.tif'
        tifffile.imwrite(fname, img_new)
    
    return img_new


##
## Add cli
##

# if __name__ == '__main__':
#     argh.dispatch_command(skeletonize_and_remove_cycle)