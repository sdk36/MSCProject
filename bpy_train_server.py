#import bpy to control the blender environment through the api
import bpy
from bpy import context

#import os to control the system
import os

#import torch and functionality for models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch import autograd
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import torch.nn.utils.weight_norm as weightNorm

#import numpy to deal with arrays
import numpy as np

#import random for random selection
import random

#import PIL for image control
import PIL

#import time for timestamp
import time

#import scipy and io for image control
import scipy.misc
from io import BytesIO

#import tensorboard for data collection
import tensorboardX as tb
from tensorboardX.summary import Summary

#import platform to view which system is running
import platform
pltform = platform.system()

#import matplotlib and functionality for image saving of plotted voxels
import matplotlib
if pltform == 'Linux':
    # Force matplotlib to not use any Xwindows backend if on linux system
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#import pickle to dump image data into 
import pickle


print("Starting Training Script" + "\n===============================================")

"""

Initial startup setup of Blender Environment

"""
def resetBlender():
    # Clear all objects
    try:
        bpy.ops.object.mode_set(mode='OBJECT') # set the mode to object
    except:
        pass
    for obj in bpy.context.scene.objects: # select all objects in the scene
        obj.select_set(True)
    bpy.ops.object.delete() # delete all the object currently present

    # Create basic ico_sphere with 7 subdivisions
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=7, location=(0.5,0.5,0.5), radius=0.5)
    bpy.ops.object.mode_set(mode='SCULPT')
    bpy.data.objects[0].select_set(True)
    bpy.context.scene.tool_settings.sculpt.use_symmetry_x = False

""""

Hyperparameter dictionaries

"""

training = dict(
        #training warmup training number
        warmup = 400,
        #discount factor
        discount = 0.95**5,
        #minibatch size
        batch_size = 96,
        #replay memory size
        rmsize = 800,
        #concurrent environment size
        env_batch = 96,
        #moving average for target network
        tau = 0.001,
        #max length for episode
        max_step = 40,
        #noise level for parameter space noise
        noise_factor = 0,
        #how many episodes to perform for validation
        validate_interval = 50,
        #how many episodes to perform for validation
        validate_episodes = 5,
        #total train times
        train_times = 2000000,
        #train times for each episode
        episode_train_times = 10,
        #resuming model path for testing
        resume = None,
        #resuming model path for testing
        #output = '{}/model'.form,
        #print out helpful info 
        debug = True,
        #random seed
        seed = 1234
)
cwd = os.getcwd()

path = dict(
        root = '{}'.format(cwd),
        #resuming model path for testing
        output = '{}/model'.format(cwd),
        #dataset path
        dataset_path = '{}/ShapeNetCore.v2'.format(cwd),
        #filewriter path
        fw_path = '{}/train_log'.format(cwd),
        #modeldir
        modeldir = '{}/modeldir/models.npy'.format(cwd),
        #pt output folder
        pt_path = '{}/pickles'.format(cwd)

)

"""

Hyperparameters

"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dset_loaders = None

#define the coordinate system.
coord = torch.zeros([1, 3, 128, 128, 128])
for i in range(128):
    for j in range(128):
        for k in range(128):
            coord[0, 0, i, j, k] = i / 127.
            coord[0, 1, i, j, k] = j / 127.
            coord[0, 2, i, j, k] = k / 127.
coord = coord.to(device)

print("\nHyperparameters saved")

#change over the criterion into Binary Cross Entropy
criterion = nn.MSELoss()

""""

Tensorboard

"""
class TensorBoard(object):
    def __init__(self, model_dir=None):
        self.summary_writer = tb.FileWriter(model_dir)

    def add_image(self, tag, img, step):
        summary = Summary()
        bio = BytesIO()

        if type(img) == str:
            img = PIL.Image.open(img)
        elif type(img) == PIL.Image.Image:
            pass
        else:
            img = scipy.misc.toimage(img)

        img.save(bio, format="png")
        image_summary = Summary.Image(encoded_image_string=bio.getvalue())
        summary.value.add(tag=tag, image=image_summary)
        self.summary_writer.add_summary(summary, global_step=step)

    def add_scalar(self, tag, value, step):
        summary = Summary(value=[Summary.Value(tag=tag, simple_value=value)])
        self.summary_writer.add_summary(summary, global_step=step)

print("\nTensorboard class set")

"""

Utilities

"""
USE_CUDA = torch.cuda.is_available()

def prRed(prt): print("\033[91m {}\033[00m" .format(prt))
def prGreen(prt): print("\033[92m {}\033[00m" .format(prt))
def prYellow(prt): print("\033[93m {}\033[00m" .format(prt))
def prLightPurple(prt): print("\033[94m {}\033[00m" .format(prt))
def prPurple(prt): print("\033[95m {}\033[00m" .format(prt))
def prCyan(prt): print("\033[96m {}\033[00m" .format(prt))
def prLightGray(prt): print("\033[97m {}\033[00m" .format(prt))
def prBlack(prt): print("\033[98m {}\033[00m" .format(prt))

def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()

def to_tensor(ndarray, device):
    return torch.tensor(ndarray, dtype=torch.float, device=device)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target, source):
    for m1, m2 in zip(target.modules(), source.modules()):
        m1._buffers = m2._buffers.copy()
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir

print("\nUtility functions loaded")

"""

Actor Model 

"""

class Actor(torch.nn.Module):
    def __init__(self, cube_len):
        super(Actor, self).__init__()
        self.cube_len = cube_len
        self.leaky_value = 0.2

        padd = (0,0,0)
        if self.cube_len == 32:
            padd = (1,1,1)

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(6, self.cube_len, kernel_size=4, stride=4, bias=False, padding=padd),
            torch.nn.BatchNorm3d(self.cube_len),
            torch.nn.LeakyReLU(self.leaky_value)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(self.cube_len, self.cube_len*2, kernel_size=4, stride=4, bias=False, padding=padd),
            torch.nn.BatchNorm3d(self.cube_len*2),
            torch.nn.LeakyReLU(self.leaky_value)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(self.cube_len*2, self.cube_len*4, kernel_size=4, stride=4, bias=False, padding=padd),
            torch.nn.BatchNorm3d(self.cube_len*4),
            torch.nn.LeakyReLU(self.leaky_value)
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(self.cube_len*4, self.cube_len*8, kernel_size=4, stride=4, bias=False, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len*8),
            torch.nn.LeakyReLU(self.leaky_value)
        )
        # self.layer5 = torch.nn.Sequential(
        #     torch.nn.Conv3d(self.cube_len*8, self.cube_len*16, kernel_size=4, stride=2, bias=False, padding=(1,1,1)),
        #     torch.nn.BatchNorm3d(self.cube_len*16),
        #     torch.nn.LeakyReLU(self.leaky_value)
        # )
        # self.layer6 = torch.nn.Sequential(
        #     torch.nn.Conv3d(self.cube_len*16, self.cube_len*32, kernel_size=4, stride=2, bias=False, padding=(1,1,1)),
        #     torch.nn.BatchNorm3d(self.cube_len*32),
        #     torch.nn.LeakyReLU(self.leaky_value)
        # )
        # self.layer7 = torch.nn.Sequential(
        #     torch.nn.Conv3d(self.cube_len*32, self.cube_len*64, kernel_size=4, stride=2, bias=False, padding=padd),
        #     torch.nn.BatchNorm3d(self.cube_len*64),
        #     torch.nn.LeakyReLU(self.leaky_value)
        # )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.cube_len*8, 19)
        )

    def forward(self, x):
        out = self.layer1(x)
        #print(out.size())  # torch.Size([100, 64, 32, 32, 32])
        out = self.layer2(out)
        #print(out.size())  # torch.Size([100, 128, 16, 16, 16])
        out = self.layer3(out)
        #print(out.size())  # torch.Size([100, 256, 8, 8, 8])
        out = self.layer4(out)
        #print(out.size())  # torch.Size([100, 512, 4, 4, 4])
        # out = self.layer5(out)
        # print(out.size())  # torch.Size([100, 200, 1, 1, 1])
        # out = self.layer6(out)
        # print(out.size())  # torch.Size([100, 200, 1, 1, 1])
        # out = self.layer7(out)
        # print(out.size())  # torch.Size([100, 200, 1, 1, 1])
        # out = F.avg_pool3d(out, 1)
        # print(out.size())
        out = out.view(out.size(0), -1)
        #print(out.size())
        out = self.fc(out)
        #print(out.size())
        out = torch.sigmoid(out)
        #print(out.size())
        print("Actor Run")
        return out

"""

Critic Model

"""

class Critic(torch.nn.Module):
    def __init__(self, cube_len):
        super(Critic, self).__init__()
        self.cube_len = cube_len
        self.leaky_value = 0.2

        padd = (0,0,0)
        if self.cube_len == 32:
            padd = (1,1,1)

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(7, self.cube_len, kernel_size=4, stride=4, bias=False, padding=padd),
            torch.nn.BatchNorm3d(self.cube_len),
            torch.nn.LeakyReLU(self.leaky_value)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(self.cube_len, self.cube_len*2, kernel_size=4, stride=4, bias=False, padding=padd),
            torch.nn.BatchNorm3d(self.cube_len*2),
            torch.nn.LeakyReLU(self.leaky_value)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(self.cube_len*2, self.cube_len*4, kernel_size=4, stride=2, bias=False, padding=padd),
            torch.nn.BatchNorm3d(self.cube_len*4),
            torch.nn.LeakyReLU(self.leaky_value)
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(self.cube_len*4, self.cube_len*8, kernel_size=4, stride=2, bias=False, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len*8),
            torch.nn.LeakyReLU(self.leaky_value)
        )
        # self.layer5 = torch.nn.Sequential(
        #     torch.nn.Conv3d(self.cube_len*8, self.cube_len*16, kernel_size=4, stride=2, bias=False, padding=(1,1,1)),
        #     torch.nn.BatchNorm3d(self.cube_len*16),
        #     torch.nn.LeakyReLU(self.leaky_value)
        # )
        # self.layer6 = torch.nn.Sequential(
        #     torch.nn.Conv3d(self.cube_len*16, self.cube_len*32, kernel_size=4, stride=2, bias=False, padding=(1,1,1)),
        #     torch.nn.BatchNorm3d(self.cube_len*32),
        #     torch.nn.LeakyReLU(self.leaky_value)
        # )
        # self.layer7 = torch.nn.Sequential(
        #     torch.nn.Conv3d(self.cube_len*32, self.cube_len*64, kernel_size=4, stride=2, bias=False, padding=padd),
        #     torch.nn.BatchNorm3d(self.cube_len*64),
        #     torch.nn.LeakyReLU(self.leaky_value)
        # )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.cube_len*8, 1)
        )

    def forward(self, x):
        out = self.layer1(x)
        #print(out.size())  # torch.Size([100, 64, 32, 32, 32])
        out = self.layer2(out)
        #print(out.size())  # torch.Size([100, 128, 16, 16, 16])
        out = self.layer3(out)
        #print(out.size())  # torch.Size([100, 256, 8, 8, 8])
        out = self.layer4(out)
        #print(out.size())  # torch.Size([100, 512, 4, 4, 4])
        # out = self.layer5(out)
        # print(out.size())  # torch.Size([100, 200, 1, 1, 1])
        # out = self.layer6(out)
        # print(out.size())  # torch.Size([100, 200, 1, 1, 1])
        # out = self.layer7(out)
        # print(out.size())  # torch.Size([100, 200, 1, 1, 1])
        # out = F.avg_pool3d(out, 1)
        # print(out.size())
        out = out.view(out.size(0), -1)
        #print(out.size())
        out = self.fc(out)
        #print(out.size())
        out = torch.sigmoid(out)
        #print(out.size())
        print("Critic Run")
        return out

"""
Decode blender instructions

"""

# Must override context while stroke is applied
# https://github.com/christian-vorhemus/procedural-3d-image-generation/blob/master/blenderBackgroundTask.py#L54
def context_override():
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                for region in area.regions:
                    if region.type == 'WINDOW':
                        scene = bpy.context.scene
                        return window, screen, area, region, scene

def decode(action, root, batch_size, width=128): # b * (18)
    # action.size(B, 19)

    u = 1.2
    action = torch.mean(action, 0)

    x0 = action[:1]
    x0 = x0.item()
    x0 = u * x0
    y0 = action[1:2]
    y0 = y0.item()
    y0 = u * y0
    z0 = action[2:3]
    z0 = z0.item()
    z0 = u * z0

    x1 = action[3:4]
    x1 = x1.item()
    x1 = u * x1
    y1 = action[4:5]
    y1 = y1.item()
    y1 = u * y1
    z1 = action[5:6]
    z1 = z1.item()
    z1 = u * z1

    x2 = action[6:7]
    x2 = x2.item()
    x2 = u * x2
    y2 = action[7:8]
    y2 = y2.item()
    y2 = u * y2
    z2 = action[8:9]
    z2 = z2.item()
    z2 = u * z2
    
    x3 = action[9:10]
    x3 = x3.item()
    x3 = u * x3
    y3 = action[10:11]
    y3 = y3.item()
    y3 = u * y3
    z3 = action[11:12]
    z3 = z3.item()
    z3 = u * z3

    ## set the pre variables

    pre0 = action[12:13]
    pre0 = pre0.item()
    pre3 = action[13:14]
    pre3 = pre3.item()

    ## set the size variables

    siz0 = action[14:15]
    siz0 = siz0.item()
    siz3 = action[15:16]
    siz3 = siz3.item()

    ## set the positional variables
    points = []
    tmp = 1./100

    for i in range(100):
        t = i * tmp
        
        # apply bezier curve across the points to create path across the object
        x = ((1-t)**3*x0 + 3 *
        (1-t)**2 * t * x1 + 3 * (1-t) * t**2 * x2 + t**3 * x3)
        y = ((1-t)**3*y0 + 3 *
        (1-t)**2 * t * y1 + 3 * (1-t) * t**2 * y2 + t**3 * y3)
        z = ((1-t)**3*z0 + 3 *
        (1-t)**2 * t * z1 + 3 * (1-t) * t**2 * z2 + t**3 * z3)
        pre = (int) ((1-t) * pre0 + t * pre3)
        siz = (int) ((1-t) * siz0 + t * siz3)
        p = (x,y,z)
        points.append((p, pre, siz))
        #print("Point {},{},{}".format(x, y, z))
    print("Points Chosen")


    ## set addition/ subration

    add = action[16:17]
    add = add.item()

    if add <= 0.5: 
        direction = 'SUBTRACT'
    else: 
        direction = 'ADD'
    
    print("Direction: {}".format(direction))

    # set the  radius

    radius = action[17:18]
    radius = radius.item()
    print("Radius: {}".format(radius))

    # set the strength setting
    strength = action[18:19]
    strength = strength.item()
    print("Strength: {}".format(strength))
    

    strokes = []
    for i, (p, pre, siz) in enumerate(points):
        stroke = {    
        "name": "stroke",
        "mouse": (0, 0),
        "mouse_event" : (0.0, 0.0),
        "pen_flip": False,
        "is_start": True if i==0 else False,
        "location": p,
        "size": siz,
        "x_tilt" : 0,
        "y_tilt" : 0,
        "pressure": pre, 
        "time": float(i)}
        strokes.append(stroke)


    # Set brush settings
    bpy.data.scenes["Scene"].tool_settings.unified_paint_settings.use_locked_size = "SCENE"
    bpy.data.scenes["Scene"].tool_settings.unified_paint_settings.unprojected_radius = radius
    bpy.ops.paint.brush_select(sculpt_tool='DRAW', toggle=False)
    bpy.data.brushes["SculptDraw"].direction = direction
    bpy.data.brushes["SculptDraw"].strength = strength
    bpy.data.brushes["SculptDraw"].curve_preset = "SMOOTH"
    bpy.data.brushes["SculptDraw"].auto_smooth_factor = 1.0

    # Apply stroke
    window, screen, area, region, scene = context_override()


    with bpy.context.temp_override(window=window,area=area,region=region):
        bpy.ops.sculpt.brush_stroke(stroke=strokes)
    #context_override(), 

    # save the model to stl file and create binvox file
    canvas_path = os.path.join(root, 'canvas')
    name = 'canvas'
    stl_path = os.path.join(canvas_path, name + '.stl')
    binvox_path = os.path.join(canvas_path, name + '.binvox')
    if os.path.exists(stl_path):
        os.remove(stl_path)
        if os.path.exists(binvox_path):
            os.remove(binvox_path)
            print('existing removed')
        bpy.ops.export_mesh.stl(filepath=stl_path)
        os.system('cd {} && binvox -d 128 canvas.stl'.format(canvas_path))
    else:    
        try:
            bpy.ops.export_mesh.stl(filepath=stl_path)
            os.system('cd {} && binvox -d 128 canvas.stl'.format(canvas_path))
            print('new added')
        except:
            print("An exception occurred") 

    # get the binvox data and set in tensor
    with open(binvox_path, 'rb') as f:
        model = read_as_3d_array(f)

    state = torch.zeros([1, 1, width, width, width])
    state[0] = torch.tensor(model.data)
    state = state.expand(batch_size, 1, width, width, width)
    return state

"""
Binvox_rw module

"""

class Voxels(object):
    """ Holds a binvox model.
    data is either a three-dimensional numpy boolean array (dense representation)
    or a two-dimensional numpy float array (coordinate representation).

    dims, translate and scale are the model metadata.

    dims are the voxel dimensions, e.g. [32, 32, 32] for a 32x32x32 model.

    scale and translate relate the voxels to the original model coordinates.

    To translate voxel coordinates i, j, k to original coordinates x, y, z:

    x_n = (i+.5)/dims[0]
    y_n = (j+.5)/dims[1]
    z_n = (k+.5)/dims[2]
    x = scale*x_n + translate[0]
    y = scale*y_n + translate[1]
    z = scale*z_n + translate[2]

    """

    def __init__(self, data, dims, translate, scale, axis_order):
        self.data = data
        self.dims = dims
        self.translate = translate
        self.scale = scale
        assert (axis_order in ('xzy', 'xyz'))
        self.axis_order = axis_order

    def clone(self):
        data = self.data.copy()
        dims = self.dims[:]
        translate = self.translate[:]
        return Voxels(data, dims, translate, self.scale, self.axis_order)

    def write(self, fp):
        write(self, fp)

def read_header(fp):
    """ Read binvox header. Mostly meant for internal use.
    """
    line = fp.readline().strip()
    if not line.startswith(b'#binvox'):
        raise IOError('Not a binvox file')
    dims = list(map(int, fp.readline().strip().split(b' ')[1:]))
    translate = list(map(float, fp.readline().strip().split(b' ')[1:]))
    scale = list(map(float, fp.readline().strip().split(b' ')[1:]))[0]
    line = fp.readline()
    return dims, translate, scale

def read_as_3d_array(fp, fix_coords=True):
    """ Read binary binvox format as array.

    Returns the model with accompanying metadata.

    Voxels are stored in a three-dimensional numpy array, which is simple and
    direct, but may use a lot of memory for large models. (Storage requirements
    are 8*(d^3) bytes, where d is the dimensions of the binvox model. Numpy
    boolean arrays use a byte per element).

    Doesn't do any checks on input except for the '#binvox' line.
    """
    dims, translate, scale = read_header(fp)
    raw_data = np.frombuffer(fp.read(), dtype=np.uint8)
    # if just using reshape() on the raw data:
    # indexing the array as array[i,j,k], the indices map into the
    # coords as:
    # i -> x
    # j -> z
    # k -> y
    # if fix_coords is true, then data is rearranged so that
    # mapping is
    # i -> x
    # j -> y
    # k -> z
    values, counts = raw_data[::2], raw_data[1::2]
    data = np.repeat(values, counts).astype(bool)
    data = data.reshape(dims)
    if fix_coords:
        # xzy to xyz TODO the right thing
        data = np.transpose(data, (0, 2, 1))
        axis_order = 'xyz'
    else:
        axis_order = 'xzy'
    return Voxels(data, dims, translate, scale, axis_order)

def read_as_coord_array(fp, fix_coords=True):
    """ Read binary binvox format as coordinates.

    Returns binvox model with voxels in a "coordinate" representation, i.e.  an
    3 x N array where N is the number of nonzero voxels. Each column
    corresponds to a nonzero voxel and the 3 rows are the (x, z, y) coordinates
    of the voxel.  (The odd ordering is due to the way binvox format lays out
    data).  Note that coordinates refer to the binvox voxels, without any
    scaling or translation.

    Use this to save memory if your model is very sparse (mostly empty).

    Doesn't do any checks on input except for the '#binvox' line.
    """
    dims, translate, scale = read_header(fp)
    raw_data = np.frombuffer(fp.read(), dtype=np.uint8)

    values, counts = raw_data[::2], raw_data[1::2]

    sz = np.prod(dims)
    index, end_index = 0, 0
    end_indices = np.cumsum(counts)
    indices = np.concatenate(([0], end_indices[:-1])).astype(end_indices.dtype)

    values = values.astype(bool)
    indices = indices[values]
    end_indices = end_indices[values]

    nz_voxels = []
    for index, end_index in zip(indices, end_indices):
        nz_voxels.extend(range(index, end_index))
    nz_voxels = np.array(nz_voxels)
    # TODO are these dims correct?
    # according to docs,
    # index = x * wxh + z * width + y; // wxh = width * height = d * d

    x = nz_voxels / (dims[0]*dims[1])
    zwpy = nz_voxels % (dims[0]*dims[1]) # z*w + y
    z = zwpy / dims[0]
    y = zwpy % dims[0]
    if fix_coords:
        data = np.vstack((x, y, z))
        axis_order = 'xyz'
    else:
        data = np.vstack((x, z, y))
        axis_order = 'xzy'

    #return Voxels(data, dims, translate, scale, axis_order)
    return Voxels(np.ascontiguousarray(data), dims, translate, scale, axis_order)

def dense_to_sparse(voxel_data, dtype=int):
    """ From dense representation to sparse (coordinate) representation.
    No coordinate reordering.
    """
    if voxel_data.ndim!=3:
        raise ValueError('voxel_data is wrong shape; should be 3D array.')
    return np.asarray(np.nonzero(voxel_data), dtype)

def sparse_to_dense(voxel_data, dims, dtype=bool):
    if voxel_data.ndim!=2 or voxel_data.shape[0]!=3:
        raise ValueError('voxel_data is wrong shape; should be 3xN array.')
    if np.isscalar(dims):
        dims = [dims]*3
    dims = np.atleast_2d(dims).T
    # truncate to integers
    xyz = voxel_data.astype(np.int)
    # discard voxels that fall outside dims
    valid_ix = ~np.any((xyz < 0) | (xyz >= dims), 0)
    xyz = xyz[:,valid_ix]
    out = np.zeros(dims.flatten(), dtype=dtype)
    out[tuple(xyz)] = True
    return out

#def get_linear_index(x, y, z, dims):
    #""" Assuming xzy order. (y increasing fastest.
    #TODO ensure this is right when dims are not all same
    #"""
    #return x*(dims[1]*dims[2]) + z*dims[1] + y

def write(voxel_model, fp):
    """ Write binary binvox format.

    Note that when saving a model in sparse (coordinate) format, it is first
    converted to dense format.

    Doesn't check if the model is 'sane'.

    """
    if voxel_model.data.ndim==2:
        # TODO avoid conversion to dense
        dense_voxel_data = sparse_to_dense(voxel_model.data, voxel_model.dims)
    else:
        dense_voxel_data = voxel_model.data

    fp.write('#binvox 1\n')
    fp.write('dim '+' '.join(map(str, voxel_model.dims))+'\n')
    fp.write('translate '+' '.join(map(str, voxel_model.translate))+'\n')
    fp.write('scale '+str(voxel_model.scale)+'\n')
    fp.write('data\n')
    if not voxel_model.axis_order in ('xzy', 'xyz'):
        raise ValueError('Unsupported voxel model axis order')

    if voxel_model.axis_order=='xzy':
        voxels_flat = dense_voxel_data.flatten()
    elif voxel_model.axis_order=='xyz':
        voxels_flat = np.transpose(dense_voxel_data, (0, 2, 1)).flatten()

    # keep a sort of state machine for writing run length encoding
    state = voxels_flat[0]
    ctr = 0
    for c in voxels_flat:
        if c==state:
            ctr += 1
            # if ctr hits max, dump
            if ctr==255:
                fp.write(chr(state))
                fp.write(chr(ctr))
                ctr = 0
        else:
            # if switch state, dump
            fp.write(chr(state))
            fp.write(chr(ctr))
            state = c
            ctr = 1
    # flush out remainders
    if ctr > 0:
        fp.write(chr(state))
        fp.write(chr(ctr))

"""

DDPG

"""

class DDPG(object):
    def __init__(self, batch_size=64, env_batch=1, max_step=40, \
                 tau=0.001, discount=0.9, rmsize=800, \
                 writer=None, resume=None, output_path=None):

        self.max_step = max_step
        self.env_batch = env_batch
        self.batch_size = batch_size        

        self.actor = Actor(128) # target, canvas, stepnum, coordconv 1 + 1 + 1 + 3
        self.actor_target = Actor(128) # target, canvas, stepnum, coordconv 1 + 1 + 1 + 3
        self.critic = Critic(128) # newcanvas, target, canvas, stepnum, coordconv 1 + 1 + 1 + 1 + 3
        self.critic_target = Critic(128) # newcanvas, target, canvas, stepnum, coordconv 1 + 1 + 1 + 1 + 3

        self.actor_optim  = Adam(self.actor.parameters(), lr=1e-2)
        self.critic_optim  = Adam(self.critic.parameters(), lr=1e-2)

        if (resume != None):
            self.load_weights(resume)

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)
        
        # Create replay buffer
        self.memory = rpm(rmsize * max_step)

        # Hyper-parameters
        self.tau = tau
        self.discount = discount

        # Tensorboard
        self.writer = writer
        self.log = 0
        
        self.state = [None] * self.env_batch # Most recent state
        self.action = [None] * self.env_batch # Most recent action
        self.choose_device()        

    def play(self, state, target=False):
        #print("final {}".format(state.shape))
        state = torch.cat((state.float(), coord.expand(state.shape[0], 3, 128, 128, 128)), 1)
        #print("new state {}".format(state.shape))
        if target:
            return self.actor_target(state)
        else:
            return self.actor(state)

    def update_gan(self, state):
        canvas = state[:, :1]
        gt = state[:, 1:2]
        fake, real, penal = update(canvas.float(), gt.float())
        if self.log % 20 == 0:
            self.writer.add_scalar('train/gan_fake', fake, self.log)
            self.writer.add_scalar('train/gan_real', real, self.log)
            self.writer.add_scalar('train/gan_penal', penal, self.log)       
        
    def evaluate(self, state, action, target=False):
        T = state[:, 2 : 3]
        gt = state[:, 1 : 2].float()
        canvas0 = state[:, :1].float()
        canvas1 = decode(action, path['root'], training['batch_size'], 128) ### need to sort
        gan_reward = cal_reward(canvas1, gt) - cal_reward(canvas0, gt)
        # L2_reward = ((canvas0 - gt) ** 2).mean(1).mean(1).mean(1) - ((canvas1 - gt) ** 2).mean(1).mean(1).mean(1)        
        coord_ = coord.expand(state.shape[0], 3, 128, 128, 128)
        merged_state = torch.cat([canvas0, canvas1, gt, (T + 1).float() / self.max_step, coord_], 1)
        # canvas0 is not necessarily added
        if target:
            Q = self.critic_target(merged_state)
            return (Q + gan_reward), gan_reward
        else:
            Q = self.critic(merged_state)
            if self.log % 20 == 0:
                self.writer.add_scalar('train/expect_reward', Q.mean(), self.log)
                self.writer.add_scalar('train/gan_reward', gan_reward.mean(), self.log)
            return (Q + gan_reward), gan_reward
    
    def update_policy(self, lr):
        self.log += 1
        
        for param_group in self.critic_optim.param_groups:
            param_group['lr'] = lr[0]
        for param_group in self.actor_optim.param_groups:
            param_group['lr'] = lr[1]
            
        # Sample batch
        state, action, reward, \
            next_state, terminal = self.memory.sample_batch(self.batch_size, device)

        self.update_gan(next_state)
        
        with torch.no_grad():
            next_action = self.play(next_state, True)
            target_q, _ = self.evaluate(next_state, next_action, True)
            target_q = self.discount * ((1 - terminal.float()).view(-1, 1)) * target_q
                
        cur_q, step_reward = self.evaluate(state, action)
        target_q += step_reward.detach()
        
        value_loss = criterion(cur_q, target_q)
        self.critic.eval()
        self.critic.zero_grad()
        value_loss.backward(retain_graph=True)
        self.critic_optim.step()
        self.critic.train()

        action = self.play(state)
        pre_q, _ = self.evaluate(state.detach(), action)
        policy_loss = -pre_q.mean()
        self.actor.eval()
        self.actor.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.actor_optim.step()
        self.actor.train()
        
        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return -policy_loss, value_loss

    def observe(self, reward, state, done, step):
        s0 = torch.tensor(self.state, device='cpu')
        a = to_tensor(self.action, "cpu")
        r = to_tensor(reward, "cpu")
        s1 = torch.tensor(state, device='cpu')
        d = to_tensor(done.astype('float32'), "cpu")
        for i in range(self.env_batch):
            self.memory.append([s0[i], a[i], r[i], s1[i], d[i]])
        self.state = state

    def noise_action(self, noise_factor, state, action):
        noise = np.zeros(action.shape)
        for i in range(self.env_batch):
            action[i] = action[i] + np.random.normal(0, self.noise_level[i], action.shape[1:]).astype('float32')
        return np.clip(action.astype('float32'), 0, 1)
    
    def select_action(self, state, return_fix=False, noise_factor=0):
        self.eval()
        #print("state shape".format(state.shape))
        with torch.no_grad():
            action = self.play(state)
            action = to_numpy(action)
        if noise_factor > 0:        
            action = self.noise_action(noise_factor, state, action)
        self.train()
        self.action = action
        if return_fix:
            return action
        return self.action

    def reset(self, obs, factor):
        self.state = obs
        self.noise_level = np.random.uniform(0, factor, self.env_batch)

    def load_weights(self, path):
        if path is None: return
        self.actor.load_state_dict(torch.load('{}/actor.pth'.format(path)))
        self.critic.load_state_dict(torch.load('{}/critic.pth'.format(path)))
        load_gan(path)
        
    def save_model(self, path):
        self.actor.cpu()
        self.critic.cpu()
        torch.save(self.actor.state_dict(),'{}/actor.pth'.format(path))
        torch.save(self.critic.state_dict(),'{}/critic.pth'.format(path))
        save_gan(path)
        self.choose_device()

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()
    
    def train(self):
        self.actor.train()
        self.actor_target.train()
        self.critic.train()
        self.critic_target.train()
    
    def choose_device(self):
        # Decoder.to(device)
        self.actor.to(device)
        self.actor_target.to(device)
        self.critic.to(device)
        self.critic_target.to(device)

"""

EValuator

"""
class Evaluator(object):

    def __init__(self, writer):    
        self.validate_episodes = training['validate_episodes']
        self.max_step = training['max_step']
        self.env_batch = training['env_batch']
        self.writer = writer
        self.log = 0

    def __call__(self, env, policy, gt, debug=False):        
        observation = None
        for episode in range(self.validate_episodes):
            # reset at the start of episode
            observation = env.reset(gt, test=True, episode=episode)
            episode_steps = 0
            episode_reward = 0.     
            assert observation is not None            
            # start episode
            episode_reward = np.zeros(self.env_batch)
            while (episode_steps < self.max_step or not self.max_step):
                action = policy(observation)
                observation, reward, done, (step_num) = env.step(action)
                episode_reward += reward
                episode_steps += 1
                env.save_image(self.log, episode_steps)
            dist = env.get_dist()
            self.log += 1
        return episode_reward, dist
"""

Save Voxel Plot




"""
def SavePloat_Voxels(canvas, log, step, writer, imgid, id):
    # width = 128
    # canvas B * 1 * width * width * width
    # gt B * 1 * width * width * width

    canvas = torch.mean(canvas, 0) # canvas 1 * 1 * width * width * width
    canvas = to_numpy(canvas) 
    canvas = np.squeeze(canvas)

    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = False
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    z, x, y = canvas.nonzero()
    ax.scatter(x, y, z, c=z, alpha=1)
    if imgid <= 10:
        writer.add_figure('{}/{}_{}.png'.format(str(imgid), str(id), str(step)), canvas, log)
    if imgid < 50:
        writer.add_figure(str(imgid) + '/_{}.png'.format(id), canvas, log)



"""

Multi

fastenv class called to create training environment for the agent, contains
* reset method
* step method
* distance method
* save image for environment

"""


class fastenv():
    def __init__(self, 
                 max_episode_length=10, env_batch=64, \
                 writer=None):  
        self.max_episode_length = max_episode_length
        self.env_batch = env_batch
        self.env = Sculpt(self.env_batch, self.max_episode_length)
        self.env.load_data()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.writer = writer
        self.test = False
        self.log = 0

    ###### to confirm and write
    def save_image(self, log, step):

        for i in range(self.env_batch):
            if self.env.imgid <= 10:
                canvas = SavePloat_Voxels(self.env.canvas, log, step, self.writer, self.env.imgid, 'canvas')      
        if step == self.max_episode_length:
            for i in range(self.env_batch):
                if self.env.imgid < 50:
                    gt = SavePloat_Voxels(self.env.gt, log, step, self.writer, self.env.imgid, 'target') 
                    canvas = SavePloat_Voxels(self.env.canvas, log, step, self.writer, self.env.imgid, 'canvas') 
        print("Saved Image")


    def step(self, action):
        with torch.no_grad():
            ob, r, d, _ = self.env.step(torch.tensor(action).to(device))
        if d[0]:
            if not self.test:
                self.dist = self.get_dist()
                for i in range(self.env_batch):
                    self.writer.add_scalar('train/dist', self.dist[i], self.log)
                    self.log += 1
        return ob, r, d, _

    def get_dist(self):
        #### confirm the mean work
        return to_numpy((((self.env.gt.float() - self.env.canvas.float())) ** 2).mean(1).mean(1).mean(1).mean(1))
        
    def reset(self, gt, test=False, episode=0):
        self.test = test
        ob = self.env.reset(gt, self.test, episode * self.env_batch)
        return ob

"""

RPM

"""
class rpm(object):
    # replay memory
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.index = 0
        
    def append(self, obj):
        if self.size() > self.buffer_size:
            print('buffer size larger than set value, trimming...')
            self.buffer = self.buffer[(self.size() - self.buffer_size):]
        elif self.size() == self.buffer_size:
            self.buffer[self.index] = obj
            self.index += 1
            self.index %= self.buffer_size
        else:
            self.buffer.append(obj)

    def size(self):
        return len(self.buffer)

    def sample_batch(self, batch_size, device, only_state=False):
        if self.size() < batch_size:
            batch = random.sample(self.buffer, self.size())
        else:
            batch = random.sample(self.buffer, batch_size)

        if only_state:
            res = torch.stack(tuple(item[3] for item in batch), dim=0)            
            return res.to(device)
        else:
            item_count = 5
            res = []
            for i in range(5):
                k = torch.stack(tuple(item[i] for item in batch), dim=0)
                res.append(k.to(device))
            return res[0], res[1], res[2], res[3], res[4]

"""

WGAN

"""

dim = 128
LAMBDA = 10 # Gradient penalty lambda hyperparameter

class TReLU(nn.Module):
    def __init__(self):
            super(TReLU, self).__init__()
            self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.alpha.data.fill_(0)

    def forward(self, x):
        x = F.relu(x - self.alpha) + self.alpha
        return x

### confirm the structure of the descriminator
class Discriminator(torch.nn.Module):
    def __init__(self, cublen):
        super(Discriminator, self).__init__()
        self.cube_len = cublen
        self.leaky_value = 0.2

        padd = (0,0,0)
        if self.cube_len == 32:
            padd = (1,1,1)

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(2, self.cube_len, kernel_size=4, stride=4, bias=False, padding=(0,0,0)),
            torch.nn.BatchNorm3d(self.cube_len),
            torch.nn.LeakyReLU(self.leaky_value)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(self.cube_len, self.cube_len*2, kernel_size=4, stride=4, bias=False, padding=(0,0,0)),
            torch.nn.BatchNorm3d(self.cube_len*2),
            torch.nn.LeakyReLU(self.leaky_value)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(self.cube_len*2, self.cube_len*4, kernel_size=4, stride=4, bias=False, padding=(0,0,0)),
            torch.nn.BatchNorm3d(self.cube_len*4),
            torch.nn.LeakyReLU(self.leaky_value)
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(self.cube_len*4, self.cube_len*8, kernel_size=4, stride=4, bias=False, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len*8),
            torch.nn.LeakyReLU(self.leaky_value)
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv3d(self.cube_len*8, 1, kernel_size=4, stride=2, bias=False, padding=padd),
            torch.nn.Sigmoid()
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.cube_len*8, 1)
        )

    def forward(self, x):
        #print(x.size()) # torch.Size([100, 1, 64, 64, 64])
        x = self.layer1(x)
        #print(x.size())  # torch.Size([100, 64, 32, 32, 32])
        x = self.layer2(x)
        #print(x.size())  # torch.Size([100, 128, 16, 16, 16])
        x = self.layer3(x)
        #print(x.size())  # torch.Size([100, 256, 8, 8, 8])
        x = self.layer4(x)
        #print(x.size())  # torch.Size([100, 512, 4, 4, 4])
        #x = self.layer5(x)
        #print(x.size())  # torch.Size([100, 200, 1, 1, 1])
        x = x.view(x.size(0), -1)
        #print(x.size())
        x = self.fc(x)
        #print(x.size())
        x = torch.sigmoid(x)
        #print(x.size())
        print("Discriminator Run")

        return x

netD = Discriminator(128)
target_netD = Discriminator(128)
netD = netD.to(device)
target_netD = target_netD.to(device)
hard_update(target_netD, netD)

optimizerD = Adam(netD.parameters(), lr=3e-4, betas=(0.5, 0.999))
def cal_gradient_penalty(netD, real_data, fake_data, batch_size):
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous()
    alpha = alpha.view(batch_size, 2, dim, dim, dim)
    alpha = alpha.to(device)
    fake_data = fake_data.view(batch_size, 2, dim, dim, dim)
    interpolates = Variable(alpha * real_data.data + ((1 - alpha) * fake_data.data), requires_grad=True)
    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(disc_interpolates, interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def cal_reward(fake_data, real_data):
    return target_netD(torch.cat([real_data, fake_data], 1))

def save_gan(path):
    netD.cpu()
    torch.save(netD.state_dict(),'{}/wgan.pth'.format(path))
    netD.to(device)

def load_gan(path):
    netD.load_state_dict(torch.load('{}/wgan.pth'.format(path)))

def update(fake_data, real_data):
    fake_data = fake_data.detach()
    real_data = real_data.detach()
    fake = torch.cat([real_data, fake_data], 1)
    real = torch.cat([real_data, real_data], 1)
    D_real = netD(real)
    D_fake = netD(fake)
    gradient_penalty = cal_gradient_penalty(netD, real, fake, real.shape[0])
    optimizerD.zero_grad()
    D_cost = D_fake.mean() - D_real.mean() + gradient_penalty
    D_cost.backward()
    optimizerD.step()
    soft_update(target_netD, netD, 0.001)
    return D_fake.mean(), D_real.mean(), gradient_penalty


"""

Environment

"""

width = 128
canvas_area = width * width * width

img_train = []
img_test = []
train_num = 0
test_num = 0

"""

Dataloader 

"""

class ShapeNetDataset(torch.utils.data.Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader"""

    def __init__(self, root):
        """Set the path for Data.
        Args:
            root: image directory.
            transform: Tensor transformer.
        """
        self.root = root
        self.listdir = os.listdir(self.root)
        self.content = np.load(path['modeldir'])

    def __getitem__(self, index):
        modeldir = self.content[index]
        print(modeldir)
        with open(modeldir, 'rb') as f:
            model = read_as_3d_array(f)
        tmodel = torch.FloatTensor(model.data)
        tmodel = tmodel.view(1, 128, 128, 128)
        return tmodel

    def __len__(self):
        return len(self.listdir)

class Sculpt:
    def __init__(self, batch_size, max_step):
        self.batch_size = batch_size
        self.max_step = max_step
        self.action_space = (19)
        self.observation_space = (self.batch_size, width, width, width, 6)
        self.test = False
    #### data loader setup from example  
      
    def load_data(self):
        # CelebA
        global dset_loaders
        print("Load Data")
        """
        global train_num, test_num
        for i in range(200000):
            img_id = '%06d' % (i + 1)
            try:
                # img = cv2.imread('./data/img_align_celeba/' + img_id + '.jpg', cv2.IMREAD_UNCHANGED)
                # img = cv2.resize(img, (width, width))
                if i > 2000:                
                    train_num += 1
                    img_train.append(img)
                else:
                    test_num += 1
                    img_test.append(img)
            finally:
                if (i + 1) % 10000 == 0:                    
                    print('loaded {} images'.format(i + 1))
        print('finish loading data, {} training images, {} testing images'.format(str(train_num), str(test_num)))
        """

        dsets = ShapeNetDataset(path['dataset_path'])
        dset_loaders = torch.utils.data.DataLoader(dsets, training['batch_size'], drop_last=True, shuffle=True, num_workers=0)


    #potentially include in the dataloaders setting load data    
    def pre_data(self, id, test):
        if test:
            img = img_test[id]
        else:
            img = img_train[id]
        # if not test:

        img = np.asarray(img)
        return np.transpose(img, (2, 0, 1))
    
    def reset(self, gt, i, test=False, begin_num=False):
        self.test = test
        self.imgid = i
        self.gt = gt
        self.tot_reward = ((self.gt.float()) ** 2).mean(1).mean(1).mean(1).mean(1)
        self.stepnum = 0
        self.canvas = torch.zeros([self.batch_size, 1, width, width, width], dtype=torch.uint8).to(device)
        self.lastdis = self.ini_dis = self.cal_dis()
        return self.observation()
    
    def observation(self):
        # canvas B * 1 * width * width * width
        # gt B * 1 * width * width * width
        # T B * 1 * width * width * width
        ob = []
        T = torch.ones([self.batch_size, 1, width, width, width], dtype=torch.uint8) * self.stepnum
        #print("T shape {}".format(T.shape))
        #print("canvas shape {}".format(self.canvas.shape))
        #print("gt shape {}".format(self.gt.shape))
        return torch.cat((self.canvas, self.gt, T), 1) # canvas, img, T

    def cal_trans(self, s, t):
        return (s.transpose(0, 3) * t).transpose(0, 3)
    
    def step(self, action):
        self.canvas = decode(action, path['root'], training['batch_size'], 128)
        self.stepnum += 1
        ob = self.observation()
        done = (self.stepnum == self.max_step)
        reward = self.cal_reward() # np.array([0.] * self.batch_size)
        return ob.detach(), reward, np.array([done] * self.batch_size), None

    def cal_dis(self):
        return (((self.canvas.float() - self.gt.float())) ** 2).mean(1).mean(1).mean(1).mean(1) #### confirm mean function
    
    def cal_reward(self):
        dis = self.cal_dis()
        reward = (self.lastdis - dis) / (self.ini_dis + 1e-8)
        self.lastdis = dis
        return to_numpy(reward)

"""

Train script

"""

# exp = os.path.abspath('.').split('/')[-1]
writer = TensorBoard(path['fw_path'])
# './train_log/{}'.format(exp)
#os.system('ln -sf ./train_log/{} ./log'.format(exp))
#os.system('mkdir ./model')

def train(agent, env, evaluate):
    train_times = training['train_times']
    env_batch = training['env_batch']
    validate_interval = training['validate_interval']
    max_step = training['max_step']
    debug = training['debug']
    episode_train_times = training['episode_train_times']
    resume = training['resume']
    output = path['output']
    time_stamp = time.time()
    step = episode = episode_steps = 0
    tot_reward = 0.
    observation = None
    noise_factor = training['noise_factor']
    for i, x in enumerate(dset_loaders):
        resetBlender()
        print("Blender Environment Reset")
        while step <= train_times:
            step += 1
            episode_steps += 1
            # reset if it is the start of episode
            if observation is None:
                observation = env.reset(x, i)
                agent.reset(observation, noise_factor)  
                print("Get Initial Observation and reset Agent")  
            action = agent.select_action(observation, noise_factor=noise_factor)
            print("Choose Action")
            observation, reward, done, _ = env.step(action)
            print("Get Observation, Reward and Done")
            agent.observe(reward, observation, done, step)

            if (episode_steps >= max_step and max_step):
                print("Episode Steps Greater then Max Step")
                if step > training['warmup']:
                    # [optional] evaluate
                    print("Greater Than Warmup")
                    #if episode > 0 and validate_interval > 0 and episode % validate_interval == 0:
                    reward, dist = evaluate(env, agent.select_action, x, debug=debug)
                    print("Evaluated Model")
                    if debug: prRed('Step_{:07d}: mean_reward:{:.3f} mean_dist:{:.3f} var_dist:{:.3f}'.format(step - 1, np.mean(reward), np.mean(dist), np.var(dist)))
                    writer.add_scalar('validate/mean_reward', np.mean(reward), step)
                    writer.add_scalar('validate/mean_dist', np.mean(dist), step)
                    writer.add_scalar('validate/var_dist', np.var(dist), step)
                    agent.save_model(path['pt_path'])
                    print("Model Saved\n=========================================")
                train_time_interval = time.time() - time_stamp
                time_stamp = time.time()
                tot_Q = 0.
                print("Got total Q")
                tot_value_loss = 0.
                if step > training['warmup']:
                    if step < 10000 * max_step:
                        lr = (3e-4, 1e-3)
                    elif step < 20000 * max_step:
                        lr = (1e-4, 3e-4)
                    else:
                        lr = (3e-5, 1e-4)
                    for i in range(episode_train_times):
                        Q, value_loss = agent.update_policy(lr)
                        print("Updated Policy")
                        tot_Q += Q.data.cpu().numpy()
                        tot_value_loss += value_loss.data.cpu().numpy()
                    writer.add_scalar('train/critic_lr', lr[0], step)
                    writer.add_scalar('train/actor_lr', lr[1], step)
                    writer.add_scalar('train/Q', tot_Q / episode_train_times, step)
                    writer.add_scalar('train/critic_loss', tot_value_loss / episode_train_times, step)
                if debug: prBlack('#{}: steps:{} interval_time:{:.2f} train_time:{:.2f}' \
                    .format(episode, step, train_time_interval, time.time()-time_stamp)) 
                time_stamp = time.time()
                # reset
                observation = None
                episode_steps = 0
                episode += 1
    
# create the output folder and set the variable output folder
output_folder = get_output_folder(path['output'], "Sculpt") 

#set the random seed 
np.random.seed(training['seed'])
torch.manual_seed(training['seed'])
if torch.cuda.is_available(): torch.cuda.manual_seed_all(training['seed'])
random.seed(training['seed'])

#switch off deterministic algorithm utilisation and instruct to benchmark and utilise the fastest
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

#create an instance of fastenv as environment, supply max step and env batch and writer.
fenv = fastenv(training['max_step'], training['env_batch'], writer)


agent = DDPG(training['batch_size'], training['env_batch'], training['max_step'], \
                training['tau'], training['discount'], training['rmsize'], \
                writer, training['resume'], training['output'])
evaluate = Evaluator(writer)
print('observation_space', fenv.observation_space, 'action_space', fenv.action_space)
train(agent, fenv, evaluate)
