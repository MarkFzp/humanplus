import numpy as np
import torch
import os
import h5py
import pickle
import fnmatch
import cv2
from time import time
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms

import IPython
e = IPython.embed

def flatten_list(l):
    return [item for sublist in l for item in sublist]

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path_list, camera_names, norm_stats, episode_ids, episode_len, 
                 chunk_size, policy_class,width=None, height=None, normalize_resnet=False,data_aug=False,
                 observation_name=[],feature_loss=False,grayscale=False,randomize_color=False,
                 randomize_index=None,randomize_data_degree=0,randomize_data=False):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_path_list = dataset_path_list
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.episode_len = episode_len
        self.chunk_size = chunk_size
        self.cumulative_len = np.cumsum(self.episode_len)
        self.max_episode_len = max(episode_len)
        self.policy_class = policy_class
        self.transformations = None
        self.width = width
        self.height = height
        self.data_aug = data_aug
        self.feature_loss = feature_loss
        self.randomize_data = randomize_data
        self.randomize_data_radian = randomize_data_degree/180*np.pi
        self.randomize_index = randomize_index
        self.grayscale = grayscale
        self.randomize_color = randomize_color
        if self.data_aug:
            #has nothing to do with the deployment of the model 
            self.transformations = [
                transforms.ColorJitter(hue=0.5,saturation=0.5),
                # transforms.Pad(padding=[int(self.width * 0.05), int(self.height * 0.05)], padding_mode='edge'),
                # transforms.RandomCrop(size=[self.height,self.width])]
            ]
        else:
            self.transformations = None
  
        self.normalize_resnet = normalize_resnet
        if self.normalize_resnet:
            #need to normalize the image to the same mean and std as the resnet model during depolyment
            self.normalize_resnet_tf = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
        self.observation_name = observation_name
        self.__getitem__(0) # initialize self.is_sim and self.transformations
        self.is_sim = False

        
        
        
    def _locate_transition(self, index):
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(self.cumulative_len > index) # argmax returns first True index
        start_ts = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        episode_id = self.episode_ids[episode_index]
        return episode_id, start_ts

    def __getitem__(self, index):
        episode_id, start_ts = self._locate_transition(index)
        dataset_path = self.dataset_path_list[episode_id]
        if "flipped" in dataset_path:
            flipped_data = True
        else:
            flipped_data = False
        with h5py.File(dataset_path, 'r') as root:
            is_sim = False
            compressed = root.attrs.get('compress', False)
            action = root['/action'][()]
            original_action_shape = action.shape
            episode_len = original_action_shape[0]
            # get observation at start_ts only
            if len(self.observation_name) > 0:
                observation_data = []
                for name in self.observation_name:
                    if name=='imu_orn':
                        observation_data.append(root[f'/observations/{name}'][()][start_ts, :2]) # only take the first 2 elements, row and pitch
                    else:
                        observation_data.append(root[f'/observations/{name}'][start_ts])
                qpos = np.concatenate(observation_data, axis=-1)
            else:
                qpos = np.zeros([original_action_shape[0],40])[start_ts]
                
            image_dict = dict()
            if self.feature_loss:
                image_dict_future = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
                if self.feature_loss:
                    dummy_index = min(start_ts+self.chunk_size, episode_len - 1)
                    image_dict_future[cam_name] = root[f'/observations/images/{cam_name}'][dummy_index]
                    
            
            if compressed:
                for cam_name in image_dict.keys():
                    decompressed_image = cv2.imdecode(image_dict[cam_name], 1)
                    if self.width is not None and self.height is not None: 
                        decompressed_image = cv2.resize(decompressed_image, (self.width, self.height), interpolation=cv2.INTER_AREA)
                    image_dict[cam_name] = np.array(decompressed_image)
                if self.feature_loss:
                    for cam_name in image_dict_future.keys():
                        decompressed_image = cv2.imdecode(image_dict_future[cam_name], 1)
                        if self.width is not None and self.height is not None: 
                            decompressed_image = cv2.resize(decompressed_image, (self.width, self.height), interpolation=cv2.INTER_AREA)
                        image_dict_future[cam_name] = np.array(decompressed_image)
                        
            # get all actions after and including start_ts
            if is_sim:
                action = action[start_ts:]
                action_len = episode_len - start_ts
            else:
                action = action[max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        padded_action = np.zeros((self.max_episode_len, original_action_shape[1]), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(self.max_episode_len)
        is_pad[action_len:] = 1

        padded_action = padded_action[:self.chunk_size]
        is_pad = is_pad[:self.chunk_size]

        # new axis for different cameras
        all_cam_images = []
        if self.feature_loss:
            all_cam_images_future = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
            if self.feature_loss:
                all_cam_images_future.append(image_dict_future[cam_name])
                
        all_cam_images = np.stack(all_cam_images, axis=0)
        if self.feature_loss:
            all_cam_images_future = np.stack(all_cam_images_future, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        if self.feature_loss:
            image_data_future = torch.from_numpy(all_cam_images_future)
            image_data = torch.cat([image_data, image_data_future], dim=0) #just cat the images together for feature loss
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)
        image_data = image_data / 255.0
        if flipped_data:
            image_data = image_data.flip(-1)
        #     cv2.imwrite(f'check_data_aug/flipped_image_{index}0.jpg', image_data[0].permute(1,2,0).numpy()*255)
        #     cv2.imwrite(f'check_data_aug/flipped_image_{index}1.jpg', image_data[1].permute(1,2,0).numpy()*255)
        # else:
        #     cv2.imwrite(f'check_data_aug/flipped_image_{index}0.jpg', image_data[0].permute(1,2,0).numpy()*255)
        #     cv2.imwrite(f'check_data_aug/flipped_image_{index}1.jpg', image_data[1].permute(1,2,0).numpy()*255)

        
        if not self.randomize_color:
            image_data = image_data[:,[2,1,0],:, :] # BGR to RGB
        else:
            order = np.random.permutation(3)
            image_data = image_data[:,order,:,:] # randomize color
            # cv2.imwrite(f'check_data_aug/augmented_image_{index}.jpg', cv2.cvtColor(image_data[0].permute(1,2,0).numpy()*255, cv2.COLOR_RGB2BGR))
        

        if self.grayscale:
            image_data = torch.mean(image_data, dim=1, keepdim=True).repeat(1,3,1,1)
        if self.data_aug:
            for transform in self.transformations:
                image_data = transform(image_data) # apply data augmentation
                # cv2.imwrite(f'check_data_aug/augmented_image_{index}0.jpg', cv2.cvtColor(image_data[0].permute(1,2,0).numpy()*255, cv2.COLOR_RGB2BGR))
                # cv2.imwrite(f'check_data_aug/augmented_image_{index}1.jpg', cv2.cvtColor(image_data[1].permute(1,2,0).numpy()*255, cv2.COLOR_RGB2BGR))
    
            #save the image for debugging
                    
        if self.randomize_data:
            #randomize the action data
            randomize_amount = torch.rand(len(self.randomize_index))*2*self.randomize_data_radian - self.randomize_data_radian
            action_data[:,self.randomize_index] += randomize_amount
            qpos_data[self.randomize_index] += randomize_amount
            
            
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        
        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_path_list,observation_name=['qpos']):
    all_qpos_data = []
    all_action_data = []
    all_action_v_data = []
    all_episode_len = []

    for dataset_path in dataset_path_list:
        try:
            with h5py.File(dataset_path, 'r') as root:
                action = root['/action'][()]
                original_action_shape = action.shape
                try:
                    observation_data = []
                    for name in observation_name:
                        if name=='imu_orn':
                            observation_data.append(root[f'/observations/{name}'][()][..., :2]) # only take the first 2 elements, row and pitch
                        else:
                            observation_data.append(root[f'/observations/{name}'][()])
                    qpos = np.concatenate(observation_data, axis=-1)
                except:
                    qpos = np.zeros([original_action_shape[0],40])

      
        except Exception as e:
            print(f'Error loading {dataset_path} in get_norm_stats')
            print(e)
            quit()
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
        all_action_v_data.append(torch.from_numpy(action[1:, :]-action[:-1, :]))
        all_episode_len.append(len(action))
    
    all_action_data = torch.cat(all_action_data, dim=0)
    all_action_v_data = torch.cat(all_action_v_data, dim=0)
    
    # normalize action data
    action_mean = all_action_data.mean(dim=[0]).float()
    action_std = all_action_data.std(dim=[0]).float()
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping    

    action_min = all_action_data.min(dim=0).values.float()
    action_max = all_action_data.max(dim=0).values.float()
    
    all_action_v_max = all_action_v_data.max(dim=0).values.float()
    all_action_v_min = all_action_v_data.min(dim=0).values.float()
    

    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0]).float()
    qpos_std = all_qpos_data.std(dim=[0]).float()
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping
        
    eps = 0.0001
    stats = {"action_mean": action_mean.numpy(), "action_std": action_std.numpy(),
            "action_min": action_min.numpy() - eps,"action_max": action_max.numpy() + eps,
            "qpos_mean": qpos_mean.numpy(), "qpos_std": qpos_std.numpy(),
            "example_qpos": qpos,
            "all_action_v_max": all_action_v_max.numpy(),
            "all_action_v_min": all_action_v_min.numpy()}
  
    return stats, all_episode_len

def find_all_hdf5(dataset_dir, skip_mirrored_data):
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            if 'features' in filename: continue
            if skip_mirrored_data and 'mirror' in filename:
                continue
            hdf5_files.append(os.path.join(root, filename))
    print(f'Found {len(hdf5_files)} hdf5 files')
    return hdf5_files

def BatchSampler(batch_size, episode_len_l, sample_weights):
    sample_probs = np.array(sample_weights) / np.sum(sample_weights) if sample_weights is not None else None
    sum_dataset_len_l = np.cumsum([0] + [np.sum(episode_len) for episode_len in episode_len_l])
    while True:
        batch = []
        for _ in range(batch_size):
            episode_idx = np.random.choice(len(episode_len_l), p=sample_probs)
            step_idx = np.random.randint(sum_dataset_len_l[episode_idx], sum_dataset_len_l[episode_idx + 1])
            batch.append(step_idx)
        yield batch

def load_data(dataset_dir_l, name_filter, camera_names, batch_size_train, batch_size_val, 
              chunk_size, skip_mirrored_data=False, load_pretrain=False, 
              policy_class=None, stats_dir_l=None, sample_weights=None, train_ratio=0.99,
              width=None, height=None, normalize_resnet=False, data_aug=False,observation_name=[],
              feature_loss=False, grayscale=False, randomize_color=False,
              randomize_index=None,randomize_data_degree=0,randomize_data=False):
    if type(dataset_dir_l) == str:
        dataset_dir_l = [dataset_dir_l]
    dataset_path_list_list = [find_all_hdf5(dataset_dir, skip_mirrored_data) for dataset_dir in dataset_dir_l]
    num_episodes_0 = len(dataset_path_list_list[0])
    dataset_path_list = flatten_list(dataset_path_list_list)
    dataset_path_list = [n for n in dataset_path_list if name_filter(n)]
    dataset_path_list = sorted(dataset_path_list)
    num_episodes_l = [len(dataset_path_list) for dataset_path_list in dataset_path_list_list]
    num_episodes_cumsum = np.cumsum(num_episodes_l)

    # obtain train test split on dataset_dir_l[0]
    # shuffled_episode_ids_0 = np.random.permutation(num_episodes_0)
    shuffled_episode_ids_0 = np.arange(num_episodes_0)
    train_episode_ids_0 = shuffled_episode_ids_0[:int(train_ratio * num_episodes_0)]
    val_episode_ids_0 = shuffled_episode_ids_0[int(train_ratio * num_episodes_0):]
    print(f'train_episode_ids_0: {train_episode_ids_0}')
    print(f'val_episode_ids_0: {val_episode_ids_0}')
    
    train_episode_ids_l = [train_episode_ids_0] + [np.arange(num_episodes) + num_episodes_cumsum[idx] for idx, num_episodes in enumerate(num_episodes_l[1:])]
    val_episode_ids_l = [val_episode_ids_0]
    train_episode_ids = np.concatenate(train_episode_ids_l)
    val_episode_ids = np.concatenate(val_episode_ids_l)
    print(f'\n\nData from: {dataset_dir_l}\n- Train on {[len(x) for x in train_episode_ids_l]} episodes\n- Test on {[len(x) for x in val_episode_ids_l]} episodes\n\n')
    print("test on ",dataset_path_list[val_episode_ids_l[0][0]])

    # obtain normalization stats for qpos and action
    # if load_pretrain:
    #     with open(os.path.join('/home/zfu/interbotix_ws/src/act/ckpts/pretrain_all', 'dataset_stats.pkl'), 'rb') as f:
    #         norm_stats = pickle.load(f)
    #     print('Loaded pretrain dataset stats')
    _, all_episode_len = get_norm_stats(dataset_path_list,observation_name=observation_name)
    train_episode_len_l = [[all_episode_len[i] for i in train_episode_ids] for train_episode_ids in train_episode_ids_l]
    val_episode_len_l = [[all_episode_len[i] for i in val_episode_ids] for val_episode_ids in val_episode_ids_l]
    train_episode_len = flatten_list(train_episode_len_l)
    val_episode_len = flatten_list(val_episode_len_l)
    if stats_dir_l is None:
        stats_dir_l = dataset_dir_l
    elif type(stats_dir_l) == str:
        stats_dir_l = [stats_dir_l]
    norm_stats, _ = get_norm_stats(flatten_list([find_all_hdf5(stats_dir, skip_mirrored_data) for stats_dir in stats_dir_l]),observation_name=observation_name)
    print(f'Norm stats from: {stats_dir_l}')

    batch_sampler_train = BatchSampler(batch_size_train, train_episode_len_l, sample_weights)
    batch_sampler_val = BatchSampler(batch_size_val, val_episode_len_l, None)


    # construct dataset and dataloader
    train_dataset = EpisodicDataset(dataset_path_list, camera_names, norm_stats, train_episode_ids, train_episode_len, chunk_size, 
                                    policy_class, width=width, height=height,normalize_resnet=normalize_resnet,data_aug=data_aug,
                                    observation_name=observation_name,feature_loss=feature_loss,grayscale=grayscale,randomize_color=randomize_color,
                                    randomize_index=randomize_index,randomize_data_degree=randomize_data_degree,randomize_data=randomize_data)
    val_dataset = EpisodicDataset(dataset_path_list, camera_names, norm_stats, val_episode_ids, val_episode_len, chunk_size, 
                                  policy_class, width=width, height=height,normalize_resnet=normalize_resnet,data_aug=False,
                                  observation_name=observation_name,feature_loss=False,grayscale=grayscale,randomize_color=False)
    train_num_workers = 8 if train_dataset.data_aug else 8
    val_num_workers = 8 if train_dataset.data_aug else 8
    print(f'Augment images: {train_dataset.data_aug}, train_num_workers: {train_num_workers}, val_num_workers: {val_num_workers}')
    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler_train, pin_memory=True, num_workers=train_num_workers, prefetch_factor=2)
    val_dataloader = DataLoader(val_dataset, batch_sampler=batch_sampler_val, pin_memory=True, num_workers=val_num_workers, prefetch_factor=2)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim

def calibrate_linear_vel(base_action, c=None):
    if c is None:
        c = 0.0 # 0.19
    v = base_action[..., 0]
    w = base_action[..., 1]
    base_action = base_action.copy()
    base_action[..., 0] = v - c * w
    return base_action

def smooth_base_action(base_action):
    return np.stack([
        np.convolve(base_action[:, i], np.ones(5)/5, mode='same') for i in range(base_action.shape[1])
    ], axis=-1).astype(np.float32)

def preprocess_base_action(base_action):
    # base_action = calibrate_linear_vel(base_action)
    base_action = smooth_base_action(base_action)

    return base_action

def postprocess_base_action(base_action):
    linear_vel, angular_vel = base_action
    linear_vel *= 1.0
    angular_vel *= 1.0
    # angular_vel = 0
    # if np.abs(linear_vel) < 0.05:
    #     linear_vel = 0
    return np.array([linear_vel, angular_vel])

### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
