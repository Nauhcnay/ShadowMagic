# convert each shadow image to region map
import numpy as np
import cv2, os
from skimage.morphology import skeletonize, dilation
from preprocess import flat_to_fillmap, fillmap_to_color, remove_stray_in_fill
from thinning import thinning
from misc import remove_alpha
from PIL import Image
from scipy.ndimage import label
from multiprocessing import Pool

def to_skeleton(edge):
    edge[0, :] = 255
    edge[-1, :] = 255
    edge[:, 0] = 255
    edge[:, -1] = 255
    skeleton = 1.0 - dilation(edge.astype(np.float32) / 255.0)
    skeleton = skeletonize(skeleton)
    skeleton = (skeleton * 255.0).clip(0, 255).astype(np.uint8)
    return skeleton

def to_edge(shadow_color):
    # get region boundary
    Xp = np.pad(shadow_color, [[0, 1], [0, 0], [0, 0]], 'symmetric').astype(np.float32)
    Yp = np.pad(shadow_color, [[0, 0], [0, 1], [0, 0]], 'symmetric').astype(np.float32)
    X = np.sum((Xp[1:, :, :] - Xp[:-1, :, :]) ** 2.0, axis=2) ** 0.5
    Y = np.sum((Yp[:, 1:, :] - Yp[:, :-1, :]) ** 2.0, axis=2) ** 0.5
    edge = np.zeros_like(shadow_color)[:, :, 0]
    edge[X > 0] = 255
    edge[Y > 0] = 255
    edge[0, :] = 255
    edge[-1, :] = 255
    edge[:, 0] = 255
    edge[:, -1] = 255
    return edge

def to_region_map(edge, skeleton):
    field = np.random.uniform(low=0.0, high=255.0, size=edge.shape).clip(0, 255).astype(np.uint8)
    field[skeleton > 0] = 255 # set high bound
    field[edge > 0] = 0 # set low bound
    filter = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]],
        dtype=np.float32) / 5.0
    height = np.random.uniform(low=0.0, high=255.0, size=field.shape).astype(np.float32)
    # height = np.ones(field.shape).astype(np.float32)
    
    # why using this filtering? doesn't make sense to me
    for _ in range(512):
        height = cv2.filter2D(height, cv2.CV_32F, filter)
        height[skeleton > 0] = 255.0
        height[edge > 0] = 0.0
    return height.clip(0, 255).astype(np.uint8)

# code adpated from danbooRegion
def get_skeleton(shadow, flat, return_region_map = True):
    # shadow map must has the same shape as the flatting map
    assert shadow.shape == flat.shape[:2]
    shadow = ~(shadow.astype(bool))
    dkernel = np.array([[0, 1, 0],[1, 1, 1],[0, 1, 0]]).astype(np.uint8)

    # segment shadow map using flatting map
    fillmap, _ = flat_to_fillmap(flat, False)
    next_region = np.unique(fillmap).max() + 1
    for r in np.unique(fillmap):
        region = fillmap == r
        _, rs = cv2.connectedComponents(region.astype(np.uint8), connectivity = 4)
        if _ > 2:
            # we skip region 0 because it is alwasy background
            for i in range(1, _):
                fillmap[rs == i] = next_region
                next_region += 1
    # if any region is completely surounded by one single region, merge it to its neighbour
    for r in np.unique(fillmap):
        rs = fillmap == r
        broder = np.logical_xor(cv2.dilate(rs.astype(np.uint8), kernel = dkernel, iterations = 1).astype(bool), rs)
        intersected_regions = np.unique(fillmap[broder])
        if len(intersected_regions) == 1:
            fillmap[rs] = intersected_regions[0]

    # flat = fillmap_to_color(fillmap)
    # refine shadow region
    shadow_fillmap = shadow.astype(int).copy()
    next_region = shadow_fillmap.max() + 1
    for r in np.unique(fillmap):
        if r == 0: continue
        shadow_cut = np.logical_and(shadow, fillmap == r)
        if (shadow_cut == True).any():
            shadow_fillmap[shadow_cut] = next_region
            next_region += 1
    remove_stray_in_fill(shadow_fillmap)
    shadow_fillmap = thinning(shadow_fillmap)
    shadow_color = fillmap_to_color(shadow_fillmap)
    
    if return_region_map:
        edge = to_edge(shadow_color)    
        skeleton = to_skeleton(edge)
        return shadow_color, to_region_map(edge, skeleton)
    else:
        return shadow_color

def topo_compute_normal(dist):
    # gradient along x axis
    c = cv2.filter2D(dist, cv2.CV_32F, np.array([[-1, +1]]))
    # gradient along y axis
    r = cv2.filter2D(dist, cv2.CV_32F, np.array([[-1], [+1]]))
    h = np.zeros_like(c + r, dtype=np.float32) + 0.75
    normal_map = np.stack([h, r, c], axis=2)
    normal_map /= np.sum(normal_map ** 2.0, axis=2, keepdims=True) ** 0.5
    return normal_map

def get_regions(region_map):
    if len(region_map.shape) == 3:
        marker = region_map.mean(axis = -1).squeeze()
    else:
        marker = region_map.copy()
    normal = topo_compute_normal(marker.astype(np.uint8)) * 127.5 + 127.5
    marker[marker > 100] = 255
    marker[marker < 255] = 0
    labels, nil = label(marker / 255)
    water = cv2.watershed(normal.clip(0, 255).astype(np.uint8), labels.astype(np.int32)) + 2
    water = thinning(water)
    # all_region_indices = find_all(water)
    # regions = np.zeros_like(region_map, dtype=np.uint8)
    # for region_indices in all_region_indices:
    #     regions[region_indices] = np.random.randint(low=0, high=255, size=(3,)).clip(0, 255).astype(np.uint8)
    # return regions
    return water

def multi_to_regions(img):
    if os.path.exists(img.replace('flat', 'shadow_color')) and os.path.exists(img.replace('flat', 'regions')):
        return None
    print('log:\topenning:%s'%img)
    flat = np.array(Image.open(img))
    shadow = np.array(Image.open(img.replace('flat', 'shadow')))
    shadow_color, region_map = get_skeleton(shadow, flat)
    Image.fromarray(shadow_color).save(img.replace('flat', 'shadow_color'))
    Image.fromarray(region_map).save(img.replace('flat', 'regions'))
    return None

if __name__ == '__main__':
    __spec__ = None
    input_path = "../dataset/img"
    multi_args = []
    for img in os.listdir(input_path):
        if 'flat' not in img: continue
        multi_args.append((str(os.path.join(input_path, img)),))
    with Pool(64) as pool:
        pool.starmap(multi_to_regions, multi_args)
        
