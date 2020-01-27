import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy.ndimage as sci_img

def decode_seg_map_sequence(label_masks, args):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, args)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks


def decode_segmap(label_mask, args, plot=False):
    n_classes = args.num_classes
    label_colours = get_labels()

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb

def encode_segmap(mask):
    "takes the png image and crates a matrix with all the classes."
    "first a matrix is created with zeros in the same lengt and with as the png"
    "than it takes the the labels and check the coulers and put the number of the class in the matrix "

    mask = mask.astype(int)

    #mask = np.delete(mask, 3, 2) "needed for the custom dataset"
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask

def get_labels():
    return np.asarray([[0, 0, 0],         # 0 bg
                     [255, 0, 255],     # 1 door
                     [255,0,0],         # 2 blind
                     [0, 255, 0],       # 3 windowframes
                     [0, 0, 255],       # 4 wall
                     [125,0,125],       # 5 drainpipe
                     [0,255,255],       # 6 windowborder
                     [255,255,0],       # 7 chimney
                     [0,125,125],       # 8 material
                     [125,0,0],         # 9 rooftop
                     [125, 125, 125]   # 10 fent
                       ]
                    )

""""
def get_labels():
    return np.asarray([[0, 0, 0],         # 0 bg
                       [0,0,255],
                       [0,255,0],
                       [255,0,0],
                       [255,255,0]]
                        )


def get_labels():
    return np.asarray([[0, 0, 0],         # 0 bg
                       [0,0,255],
                       [255,0,0],
                       [0, 255, 0],
                       [255,255,0]]
                        )
"""
def _tile_segmap_v2(img: np.ndarray, segmap: np.ndarray, ntiles: int, tile_sz: int = 20,
                    prioritize_edges: float = 0.3):
    # calculate offset for tile center
    offset = int(tile_sz // 2)
    offset2 = int((tile_sz - 1) // 2)

    # find edges in segmap
    segmap_min = sci_img.minimum_filter(segmap, size=(3, 3))
    segmap_max = sci_img.maximum_filter(segmap, size=(3, 3))
    segmap_edge = (segmap_min != segmap_max).astype("float32")
    segmap_edge *= (segmap + 1)  # +1 allows distinction between background edges and pixels that are not on an edge

    # only consider the part of the segmap where a full-size tile will fit
    segmap_crop = segmap[offset:-offset2, offset:-offset2]
    segmap_edge = segmap_edge[offset:-offset2, offset:-offset2]

    # find coordinates of all pixels per class
    unique_labels = np.unique(segmap_crop)
    class_pixel_coords = [np.array(np.where(segmap_crop == i)).transpose((1, 0)) for i in unique_labels]
    class_edge_pixel_coords = [np.array(np.where(segmap_edge == (i + 1))).transpose((1, 0)) for i in unique_labels]

    # create tile datastructure
    img_tiles = np.empty((ntiles, img.shape[0], tile_sz, tile_sz), dtype="float32")
    segmap_tiles = np.empty((ntiles, tile_sz, tile_sz), dtype="long")
    tile_labels = np.empty((ntiles), dtype="float32")

    cls = -1
    num_cls = unique_labels.shape[0]
    for i in range(ntiles):
        # cycle through classes
        cls = (cls + 1) % num_cls
        cls_id = unique_labels[cls]

        # randomly sample from either all pixels or edge pixels
        sample_type = np.random.rand()
        if sample_type <= prioritize_edges:
            # get from edge list
            num_px_edge = class_edge_pixel_coords[cls].shape[0]
            px_id = np.random.randint(num_px_edge)
            px_coord = class_edge_pixel_coords[cls][px_id]
        else:
            # get from full list
            num_px = class_pixel_coords[cls].shape[0]
            px_id = np.random.randint(num_px)
            px_coord = class_pixel_coords[cls][px_id]

        # get start and stop coordinates of tiles
        tile_x_start = px_coord[1]  # no offset required due to slice of segmap above
        tile_x_stop = tile_x_start + tile_sz
        tile_y_start = px_coord[0]
        tile_y_stop = tile_y_start + tile_sz

        # get tile from base image, store associated class ID
        img_tiles[i] = img[:, tile_y_start:tile_y_stop, tile_x_start:tile_x_stop]
        segmap_tiles[i] = segmap[tile_y_start:tile_y_stop, tile_x_start:tile_x_stop]
        tile_labels[i] = cls_id

    return img_tiles, segmap_tiles, tile_labels