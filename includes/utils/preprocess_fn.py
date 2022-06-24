import os

def rle_decode(rleString,shape):
    height = shape[0]
    width = shape[1]
    rows,cols = height,width
    if rleString == 'nan':
        return np.zeros(rows * cols, dtype=np.uint8).reshape(cols, rows)
    else:
        rleNumbers = [int(numstring) for numstring in rleString.split(' ')]
        rlePairs = np.array(rleNumbers).reshape(-1,2)
        img = np.zeros(rows*cols,dtype=np.uint8)
        for index,length in rlePairs:
            index -= 1
            img[index:index+length] = 255
        img = img.reshape(cols,rows)
        img = img/255.0
        return img


def rle_encode(img):
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def padImage(img, shape):
    
    shape = [shape[1], shape[0]]
    o_width = img.shape[0]
    o_height = img.shape[1]
    
    h_dif = 0
    v_dif = 0
    
    if(o_width < shape[1]):
        h_dif = np.abs(shape[1] - o_width) // 2
    else:
        shape[0] = img.shape[1]
    if(o_height < shape[0]):
        v_dif = np.abs(shape[0] - o_height) // 2
    else:
        shape[1] = img.shape[0]
    
    d_img = np.zeros((shape[1], shape[0]), dtype=np.float32)
    
    d_img[v_dif:v_dif+img.shape[0], h_dif:h_dif+img.shape[1]] = img
    
    return d_img, h_dif, v_dif



def patchify(img, shape):
    patches = []
    
    for j in range(0, img.shape[1], shape[1]):
        for i in range(0, img.shape[0], shape[0]):
            
            if(i > np.abs(img.shape[1] - shape[1])):
                i = np.abs(img.shape[1] - shape[1])
            if(j > np.abs(img.shape[0] - shape[0])):
                j = np.abs(img.shape[0] - shape[0])
                
            patch = img[j:shape[1] + j, i:shape[0] + i]
            patches.append(patch)
    return np.array(patches)

def stitchPatches(patches, shape):
    
    img = np.zeros((shape[1], shape[0]))
    last_x = 0
    last_y = 0
    
    for patch in patches:
        
        if(last_x > img.shape[1] - patch.shape[0]):
            last_x = img.shape[1] - patch.shape[0]
        if(last_y > img.shape[0] - patch.shape[1]):
            last_y = img.shape[0] - patch.shape[1]
        
        img[last_y : last_y + patch.shape[1], last_x : last_x + patch.shape[0]] = patch
        
        if(last_x >= img.shape[1] - patch.shape[0]):
            last_x = 0
            last_y += patch.shape[1]
        elif(last_x < img.shape[1] - patch.shape[0]):
            last_x += patch.shape[1]
        
    return img