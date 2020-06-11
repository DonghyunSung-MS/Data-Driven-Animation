import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset

class PFNNDataSet(Dataset):
    def __init__(self, filename, args):
        database = np.load(filename)
        """loading data"""
        X = database['Xun']
        Y = database['Yun']
        P = database['Pun']

        """processing data"""
        j = args.num_joint
        w = ((args.window_size*2)//10)

        Xmean, Xstd = X.mean(axis=0), X.std(axis=0)
        Ymean, Ystd = Y.mean(axis=0), Y.std(axis=0)

        Xstd[w*0:w* 1] = Xstd[w*0:w* 1].mean() # Trajectory Past Positions
        Xstd[w*1:w* 2] = Xstd[w*1:w* 2].mean() # Trajectory Future Positions
        Xstd[w*2:w* 3] = Xstd[w*2:w* 3].mean() # Trajectory Past Directions
        Xstd[w*3:w* 4] = Xstd[w*3:w* 4].mean() # Trajectory Future Directions
        Xstd[w*4:w*10] = Xstd[w*4:w*10].mean() # Trajectory Gait

        #Mask Out Unused Joints in Input
        joint_weights = np.array([
            1,
            1e-10, 1, 1, 1, 1,
            1e-10, 1, 1, 1, 1,
            1e-10, 1, 1,
            1e-10, 1, 1,
            1e-10, 1, 1, 1, 1e-10, 1e-10, 1e-10,
            1e-10, 1, 1, 1, 1e-10, 1e-10, 1e-10]).repeat(3)

        Xstd[w*10+j*3*0:w*10+j*3*1] = Xstd[w*10+j*3*0:w*10+j*3*1].mean() / (joint_weights * 0.1) # Pos
        Xstd[w*10+j*3*1:w*10+j*3*2] = Xstd[w*10+j*3*1:w*10+j*3*2].mean() / (joint_weights * 0.1) # Vel
        Xstd[w*10+j*3*2:          ] = Xstd[w*10+j*3*2:          ].mean() # Terrain

        Ystd[0:2] = Ystd[0:2].mean() # Translational Velocity
        Ystd[2:3] = Ystd[2:3].mean() # Rotational Velocity
        Ystd[3:4] = Ystd[3:4].mean() # Change in Phase
        Ystd[4:8] = Ystd[4:8].mean() # Contacts

        Ystd[8+w*0:8+w*1] = Ystd[8+w*0:8+w*1].mean() # Trajectory Future Positions
        Ystd[8+w*1:8+w*2] = Ystd[8+w*1:8+w*2].mean() # Trajectory Future Directions

        Ystd[8+w*2+j*3*0:8+w*2+j*3*1] = Ystd[8+w*2+j*3*0:8+w*2+j*3*1].mean() # Pos
        Ystd[8+w*2+j*3*1:8+w*2+j*3*2] = Ystd[8+w*2+j*3*1:8+w*2+j*3*2].mean() # Vel
        Ystd[8+w*2+j*3*2:8+w*2+j*3*3] = Ystd[8+w*2+j*3*2:8+w*2+j*3*3].mean() # Rot
        '''
        path = filename.split('.')
        np.savez_compressed("." + path[1] + "_para.npz", Xmean = Xmean,
                                                        Xstd = Xstd,
                                                        Ymean = Ymean,
                                                        Yste = Ystd)

        '''                        
        """making tensor"""
        self.len_data = X.shape[0]
        self.x = torch.from_numpy(X).unsqueeze(1)
        self.y = torch.from_numpy(Y).unsqueeze(1)
        self.phase = torch.reshape(torch.from_numpy(P), [-1, 1, 1])

        print("x",self.x.size())
        print("y",self.y.size())
        print("p",self.phase.size())
        #print("p_dim",self.phase.dim())
    def __getitem__(self, index):
        return self.x[index], self.phase[index], self.y.data[index]

    def __len__(self):
        return self.len_data
