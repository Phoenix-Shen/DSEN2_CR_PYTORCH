import numpy as np
import rasterio
import torch as t
#获得图片的原始数据
def get_raw_data(path):
    with rasterio.open(path, driver='GTiff') as src:
        image = src.read()

    # checkimage for nans
    image[np.isnan(image)] = np.nanmean(image)

    return image.astype('float32')

#获得RGB的数据或者SAR图片数据，用于后续处理
def get_rgb_preview(r, g, b, sar_composite=False):
    if not sar_composite:

        # stack and move to zero
        rgb = np.dstack((r, g, b))
        rgb = rgb - np.nanmin(rgb)

        # treat saturated images, scale values
        if np.nanmax(rgb) == 0 :
            rgb = 255 * np.ones_like(rgb)
        else:
            rgb = 255 * (rgb / np.nanmax(rgb))

        # replace nan values before final conversion
        rgb[np.isnan(rgb)] = np.nanmean(rgb)

        return rgb.astype(np.uint8)

    else:
        # generate SAR composite
        HH = r
        HV = g

        HH = np.clip(HH, -25.0, 0)
        HH = (HH + 25.1) * 255 / 25.1
        HV = np.clip(HV, -32.5, 0)
        HV = (HV + 32.6) * 255 / 32.6

        rgb = np.dstack((np.zeros_like(HH), HH, HV))

        return rgb.astype(np.uint8)

#CARL LOSS (critical)
def carl_error(y_true,csm, y_pred,input_cloudy):
    """Computes the Cloud-Adaptive Regularized Loss (CARL)"""
    
    clearmask = t.ones_like(csm) - csm
    predicted = y_pred
    # input_cloudy = y_pred
    target = y_true

    cscmae = t.mean(clearmask * t.abs(predicted - input_cloudy) + csm * t.abs(
        predicted - target)) + 1.0 * t.mean(t.abs(predicted - target))
    
    return cscmae
