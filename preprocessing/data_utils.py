import struct
import numpy as np
from PIL import Image
 
def read_record_ETL8B2(f):
    s = f.read(512)
    r = struct.unpack('>2H4s504s', s)
    i1 = Image.frombytes('1', (64, 63), r[3], 'raw')
    return r + (i1,)

def get_ETL8B_data(dataset, categories, samplesPerCategory, vectorize=False, resize=None):
    W, H = 64, 64
    new_img = Image.new('1', (W, H))

    
    filename = '../ETLC/ETL8B/ETL8B2C'+str(dataset)

    X = []
    Y = []

    try:
        iter(categories)
    except:
        categories = [categories]

    for id_category in categories:
        with open(filename, 'r') as f:
            f.seek((id_category * 160 + 1) * 512)
            for i in range(samplesPerCategory):
                try:
                    r = read_record_ETL8B2(f)
                    new_img.paste(r[-1], (0,0))    
                    iI = Image.eval(new_img, lambda x: not x)

                    if resize:
                        # new_img.thumbnail(resize, Image.ANTIALIAS)
                        new_img.thumbnail(resize)
                        shapes = resize[0], resize[1]
                    else:
                        shapes = W, H
                    
                    if vectorize:
                        outData = np.asarray(new_img.getdata()).reshape(shapes[0]*shapes[1])
                    else:
                        outData = np.asarray(new_img.getdata()).reshape(shapes[0],shapes[1])
                    
                    X.append(outData)
                    Y.append(r[1])
                except:
                    break
    return np.asarray(X, dtype=np.int32), np.asarray(Y, dtype=np.int32)