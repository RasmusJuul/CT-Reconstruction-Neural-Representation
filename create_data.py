import numpy as np
import scipy as sp
from joblib import Parallel, delayed
from tqdm import tqdm
import tifffile
import os

def generate_points(num_points, x_range, y_range, min_dist,rng):
    #generate vector points for the starting positions of every fibre.
    points = []
    num_tries = 0
    while len(points) < num_points:
        new_point = np.array([rng.uniform(x_range[0], x_range[1]), 
                              rng.uniform(y_range[0], y_range[1])])
        if all(np.linalg.norm(new_point - point) >= min_dist for point in points):
            points.append(new_point)
        num_tries += 1
        if num_tries > 5000:
            print("failed after 5000 tries")
            break
        
        
    return np.array(points)
    
#create a single cylinder. This is called in the below function
def create_cylinder(volume_size, radius, length, orientation, start_point):
    # Initialize the volume with zeros
    volume = np.zeros(volume_size, dtype=int)
    
    # Ensure orientation is a numpy array of floats
    orientation = np.array(orientation, dtype=float)
    
    # Ensure start_point is a numpy array of integers
    start_point = np.array(start_point, dtype=int)
    
    # Ensure radius and length are integers
    radius = int(radius)
    length = int(length)
    
    # Generate a grid of points
    x = np.arange(volume_size[0])
    y = np.arange(volume_size[1])
    z = np.arange(volume_size[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Shift grid to align with start_point
    X -= start_point[0]
    Y -= start_point[1]
    Z -= start_point[2]
    
    # Calculate distance from each point to the axis of the cylinder
    distances = np.sqrt((Y * orientation[2] - Z * orientation[1])**2 +
                        (Z * orientation[0] - X * orientation[2])**2 + 
                        (X * orientation[1] - Y * orientation[0])**2)
    # Set points inside the cylinder to 1
    volume[(distances <= radius)] = 1
    
    return volume



#create a bundle of fibres by calling the above function in a loop
def create_fibre_bundle(volume_size,radius,length,n_fibres,misalignment,rng):

    nvs = (volume_size[0],volume_size[1],30)
    volume_out = np.zeros(nvs,dtype=np.float32)
    points_out = generate_points(n_fibres, (30,volume_size[0]*0.9), (30,volume_size[0]*0.9), radius*3, rng)
    sp1 = points_out[:,0].reshape(-1,1)
    sp2 = points_out[:,1].reshape(-1,1)

    sp3 = np.zeros((sp2.shape[0],1))
    start_points = np.concatenate((sp1,sp2,sp3),1)
    
    
    for n1 in range(sp2.shape[0]):
        orientation = np.clip((0,0,1) + (rng.rand(3)*misalignment-misalignment/2),-1,1)
        this_start_points = start_points[n1,:]
        
        this_cylinder_volume = create_cylinder(nvs, radius, length, orientation, this_start_points)
        volume_out+= this_cylinder_volume
    volume_out = np.clip(volume_out,0,1)
    return volume_out


def save_file(mode,folder_name,count,i,vol):
    num_subfolders = max(1,int(count/100))
    subfolder = str(i%num_subfolders).zfill(3)
    
    tifffile.imwrite(f"data/{folder_name}/{mode}/{subfolder}/fiber_{str(i).zfill(5)}.tif",(vol*255).astype("uint8"))

def main(mode,folder_name,count,i,seed):
    rng = np.random.RandomState(seed=seed)
    vol_shape_zxy = np.array([300, 300, 300])
    vol_shape_yxz = np.flip(vol_shape_zxy)
    # Example usage:
    radius = 6 #in um
    length = 400
    orientation = (0, 0, 1) #(x,y,z)
    start_point = (0, 0, 0)
    misalignment = rng.rand(1)[0] #0-1
    n_fibres = rng.randint(75,125)
    #create 
    #if making training data in a loop, start here (remove random seed for different volumes)
    cylinder_volume = create_fibre_bundle(vol_shape_zxy,radius,length,n_fibres,misalignment,rng)
    cylinder_volume = sp.ndimage.zoom(cylinder_volume,(1,1,10),order=1)
    vol = np.ascontiguousarray(np.transpose(cylinder_volume,(2,1,0)))

    save_file(mode,folder_name,count,i,vol)
    


if __name__=="__main__":
    folder_name = "synthetic_fibers"
    
    os.makedirs(f"data/{folder_name}/train", exist_ok=True)
    os.makedirs(f"data/{folder_name}/test", exist_ok=True)
    os.makedirs(f"data/{folder_name}/validation", exist_ok=True)

    count_train = 2
    count_test = 2
    count_validation = 2

    for i in range(max(1,int(count_train / 100))):
        os.makedirs(f"data/{folder_name}/train/{str(i).zfill(3)}", exist_ok=True)
    for i in range(max(1,int(count_test / 100))):
        os.makedirs(f"data/{folder_name}/test/{str(i).zfill(3)}", exist_ok=True)
    for i in range(max(1,int(count_validation / 100))):
        os.makedirs(f"data/{folder_name}/validation/{str(i).zfill(3)}", exist_ok=True)

    rng_train = np.random.SeedSequence(42).generate_state(count_train)
    rng_test = np.random.SeedSequence(1337).generate_state(count_test)
    rng_validation = np.random.SeedSequence(1997).generate_state(count_validation)


    Parallel(n_jobs=-1)(delayed(main)
                        ("train",folder_name,count_train,i,rng_train[i])
                        for i in tqdm(range(count_train), unit="image", desc="creating training fiber volumes"))
    Parallel(n_jobs=-1)(delayed(main)
                        ("test",folder_name,count_test,i,rng_test[i])
                        for i in tqdm(range(count_test), unit="image", desc="creating testing fiber volumes"))
    Parallel(n_jobs=-1)(delayed(main)
                        ("validation",folder_name,count_validation,i,rng_validation[i])
                        for i in tqdm(range(count_validation), unit="image", desc="creating validation fiber volumes"))

























