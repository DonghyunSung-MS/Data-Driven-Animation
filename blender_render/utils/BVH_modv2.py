import re
import numpy as np

from Animation import Animation
#from pyquaternion import Quaternion
from Quaternions import Quaternions

channelmap = {
    'Xrotation': 'x',
    'Yrotation': 'y',
    'Zrotation': 'z'
}
channelmap_inv = {
    'x' : 'Xrotation',
    'y' : 'Yrotation',
    'z' : 'Zrotation'
}

ordermap = {
    'x': 0,
    'y': 1,
    'z': 2
}

def load(filename, start=None, end=None, order=None, world=False):
    """
    Reads a BVH file and constructs an animation

    Parameters
    ----------
    filename: str
        File to be opened

    start : int
        Optional Starting Frame

    end : int
        Optional Ending Frame

    order : str
        Optional Specifier for joint order.
        Given as string E.G 'xyz', 'zxy'

    world : bool
        If set to true euler angles are applied
        together in world space rather than local
        space

    Returns
    -------

    (animation, joint_names, frametime)
        Tuple of loaded animation and joint names
    """

    f = open(filename, "r")

    i = 0
    active = -1
    end_site = False

    names = []
    orients = Quaternions.id(0)
    offsets = np.array([]).reshape((0,3))
    parents = np.array([],dtype=int)
    channels_list = np.array([],dtype=int)
    channels_rot_map = []

    for line in f:
        if "HIERARCHY" in line : continue
        if "MOTION" in line : continue

        """ Modified line read to handle mixamo data """
        rmatch = re.match(r"ROOT (\w+:?\w+)", line)
        if rmatch:
            names.append(rmatch.group(1))
            offsets    = np.append(offsets,    np.array([[0,0,0]]),   axis=0)
            orients.qs = np.append(orients.qs, np.array([[1,0,0,0]]), axis=0)
            parents    = np.append(parents, active)
            active = (len(parents)-1)
            continue
        if "{" in line: continue

        if "}" in line:
            if end_site: end_site = False
            else: active = parents[active]
            continue
        offmatch = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
        if offmatch:
            if not end_site:
                offsets[active] = np.array([list(map(float, offmatch.groups()))])
            continue

        chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
        if chanmatch:
            channels = int(chanmatch.group(1))
            channels_list = np.append(channels_list, channels)
            if channels == 1:
                parts = line.split()
                channels_rot_map.append(channelmap[parts[-1]])
            else:
                channels_rot_map.append(None)

            if order is None:
                channelis = 0 if channels == 3 else 3
                channelie = 3 if channels == 3 else 6
                parts = line.split()[2+channelis:2+channelie]
                if any([p not in channelmap for p in parts]):
                    continue
                order = "".join([channelmap[p] for p in parts])
            continue
        jmatch = re.match("\s*JOINT\s+(\w+:?\w+)", line)
        if jmatch:
            names.append(jmatch.group(1))
            offsets    = np.append(offsets,    np.array([[0,0,0]]),   axis=0)
            orients.qs = np.append(orients.qs, np.array([[1,0,0,0]]), axis=0)
            parents    = np.append(parents, active)
            active = (len(parents)-1)
            continue

        if "End Site" in line:
            end_site = True
            continue

        fmatch = re.match("\s*Frames:\s+(\d+)", line)
        if fmatch:
            if start and end:
                fnum = (end - start)-1
            else:
                fnum = int(fmatch.group(1))
            jnum = len(parents)
            positions = offsets[np.newaxis].repeat(fnum, axis=0)
            rotations = np.zeros((fnum, len(orients), 3))
            continue

        fmatch = re.match("\s*Frame Time:\s+([\d\.]+)", line)
        if fmatch:
            frametime = float(fmatch.group(1))
            continue

        if (start and end) and (i < start or i >= end-1):
            i += 1
            continue
        #print(channels_list)
        #print(channels_rot_map)
        dmatch = line.strip().split(' ')
        if dmatch:
            data_block = np.array(list(map(float, dmatch)))
            N = len(parents) #skeleton

            fi = i - start if start else i
            #print(fi)
            if channels == 3 or channels == 1:
                positions[fi,0:1] = data_block[0:3]
                start_data = 3
                for index in range(N):
                    if channels_list[index] == 1: #revolute to zyx order mapping
                        rev = data_block[start_data : start_data+1][0]
                        #print(index, channels_rot_map[index], rev)
                        if channels_rot_map[index] == 'z':
                            rotations[fi,  index, :] = np.array([rev, 0, 0]).reshape(1,3)
                        elif channels_rot_map[index] == 'y':
                            rotations[fi, index,  :] = np.array([0, rev, 0]).reshape(1,3)
                        elif channels_rot_map[index] == 'x':
                            rotations[fi, index,  :] = np.array([0, 0, rev]).reshape(1,3)
                        start_data += 1
                    else:
                        #print(data_block[start_data : start_data + 3].reshape(1,3))
                        rotations[fi, index, :] = data_block[start_data : start_data + 3].reshape(1,3)
                        start_data += 3
            elif channels == 6:
                #print("here")
                data_block = data_block.reshape(N,6)
                positions[fi,:] = data_block[:,0:3]
                rotations[fi,:] = data_block[:,3:6]
            elif channels == 9:
                positions[fi,0] = data_block[0:3]
                data_block = data_block[3:].reshape(N-1,9)
                rotations[fi,1:] = data_block[:,3:6]
                positions[fi,1:] += data_block[:,0:3] * data_block[:,6:9]
            else:
                raise Exception("Too many channels! %i" % channels)
            print()
            i += 1

    f.close()
    print()
    print("Reading END")
    print()
    print(names)
    print(parents)
    print(channels_list)
    print(channels_rot_map) #None menas spherical
    #print(rotations.shape)
    #print(rotations)
    rotations = Quaternions.from_euler(np.radians(rotations), order=order, world=world)
    #print(rotations.qs)

    return (Animation(rotations, positions, orients, offsets, parents), names, frametime)

if __name__=="__main__":
    load('./example0.bvh')
