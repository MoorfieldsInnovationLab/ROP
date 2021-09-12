from queue import Queue
from threading import Thread
import cv2
import numpy as np

def get_transform_matrix(in_size, out_size, rotate, scale, translate):
    # center to top left corner
    h, w = in_size
    cy = h / 2
    cx = w / 2
    C1 = np.array([
        [1, 0, -cx],
        [0, 1, -cy],
        [0, 0, 1]
    ], dtype=np.float) 
    
    # rotate
    th = rotate * np.pi / 180
    R = np.array([
        [np.cos(th), -np.sin(th), 0],
        [np.sin(th), np.cos(th), 0],
        [0, 0, 1]
    ], dtype=np.float)
    
    # scale
    sx, sy = scale
    S = np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ], dtype=np.float) 
    # top left corner to center
    h, w = out_size
    ty = h / 2
    tx = w / 2
    C2 = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=np.float) 
    # translate
    ty, tx = translate
    T = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=np.float) 
    
    return T @ C2 @ S @ R @ C1

max_rotation = 30
max_scale = 1.1
max_tx = 64
max_ty = 64

patch_size = 512, 512

max_brightness = 0.1
max_contrast = 1.1
max_gamma = 1.5

def intensity_transform(x):
    b = np.random.uniform(-max_brightness, max_brightness)
    c = np.random.uniform(1, max_contrast)
    if np.random.choice([True, False]):
        c = 1 / c
    g = np.random.uniform(1, max_gamma)
    if np.random.choice([True, False]):
        g = 1 / g

    return c * x ** g + b

    
def sample_matrix(in_size, patch_size, scaling_factor):
    
    rot = np.random.uniform(-max_rotation, max_rotation)
    scale = np.random.uniform(1, max_scale)
    if np.random.choice([True, False]):
        scale = 1 / scale
    tx = np.random.uniform(-max_tx, max_tx)
    ty = np.random.uniform(-max_ty, max_ty)

    sx, sy = scaling_factor
    
    M = get_transform_matrix(in_size, patch_size, rot, (sx * scale, sy * scale), (tx, ty))
    return M

def sample_patch(image, augment=True):
    in_size = image.shape[:2]
    
    sf = 512 / 1200 # rescale 1200 pixels to 512
    if augment:
        M = sample_matrix(in_size, patch_size=patch_size, scaling_factor = (sf, sf) )
        x = cv2.warpAffine(image, M[:2], dsize=patch_size) / 255
        x = intensity_transform(x)

        # horizontal flip
        if np.random.choice([True, False]):
            x = x[:,::-1]
    else:
        M = get_transform_matrix(in_size, patch_size, 0, (sf, sf), (0, 0))
        x = cv2.warpAffine(image, M[:2], dsize=patch_size) / 255

    # scale to range of normalized pre-trained images
    x = 4 * x - 2
    return x


class BatchPreparationClassBalanced:
    
    
    def __init__(self, grading_labels, grading_data, n_channels=1, batch_size=16):
        
        self.grading_labels = grading_labels
        self.grading_data = grading_data
        
        self.n_classes = len(self.grading_data)
        self.n_channels = n_channels
        self.batch_size = batch_size
        
        self.queues = [(grading, self.get_queue(images)) 
                       for grading, images in grading_data.items()
                       if len(images) # some folds don't have all classes
                      ]
        

    def get_queue(self, images):
        
        queue = Queue(maxsize=10)
        def prepare_batches():       
            
            while True:
                image = images[np.random.randint(len(images))]
                x = sample_patch(image)
                queue.put(x)

        preparation_thread = Thread(target=prepare_batches, daemon=True)
        preparation_thread.start()

        return queue

    def get_batch(self):
        
        x_in = np.zeros((self.batch_size, *patch_size, self.n_channels))
        y_true = np.zeros((self.batch_size, self.n_classes))
        
        for i in range(self.batch_size):
            grading, queue = self.queues[i % len(self.queues)]
            
            x_in[i] = queue.get()
            y_true[i] = self.grading_labels[grading]

        return x_in, y_true

