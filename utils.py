import numpy as np
import rasterio
import os

def cloud_mask(cloud_band):
    cloud = (cloud_band >> 3) & 1
    dilated = (cloud_band >> 1) & 1
    return cloud | dilated

def calculate_uhi(tiff_path):
    if os.path.exists(tiff_path):
        with rasterio.open(tiff_path) as src:
            band10 = src.read(1).astype(float)  # LST band
            cloud = src.read(3).astype(int)     # Cloud mask band
            unc = src.read(2).astype(float) * 0.01  # Uncertainty band

            # Apply cloud mask
            mask = cloud_mask(cloud)

           
            LST_K = band10 * 0.00341802 + 149
            LST_C = LST_K - 273.15

            # Apply masking
            lst2 = np.copy(LST_C)
            lst2[(mask == 1)] = np.nan  
            nan_count=np.isnan(lst2).sum()
            if nan_count>= 0.2* lst2.shape[0]*lst2.shape[1]:
                return (np.nan,np.nan)
            
            mean = np.nanmean(lst2)
            sd = np.nanstd(lst2)

            # Initialize UHI array
            uhi = np.zeros(lst2.shape)

            avg_urban = 0
            avg_rural = 0
            urban_count = 0
            rural_count = 0
            # Classify urban and rural pixels
            for i in range(lst2.shape[0]):
                for j in range(lst2.shape[1]):
                    if np.isnan(lst2[i][j]):
                        continue  

                    if lst2[i][j] > mean + 0.5 * sd:
                        uhi[i][j] = 1  # Urban
                        avg_urban += lst2[i][j]
                        urban_count += 1
                    else:
                        uhi[i][j] = -1  # Rural
                        avg_rural += lst2[i][j]
                        rural_count += 1


            if urban_count > 0:
                avg_urban /= urban_count
            else:
                avg_urban = np.nan  

            if rural_count > 0:
                avg_rural /= rural_count
            else:
                avg_rural = np.nan  

            if not np.isnan(avg_urban) and not np.isnan(avg_rural):
                uhi_intensity = avg_urban - avg_rural
                uhi_ratio = uhi_intensity / avg_rural if avg_rural != 0 else np.nan
            else:
                uhi_intensity = np.nan
                uhi_ratio = np.nan

            return (uhi_intensity, uhi_ratio)
    else:
        return (np.nan,np.nan)

def get_features(tiff_path,tiff_path2):
    if os.path.exists(tiff_path) and os.path.exists(tiff_path2):
        with rasterio.open(tiff_path) as src1, rasterio.open(tiff_path2) as src2:

            band10 = src1.read(1).astype(float)  # LST band
            cloud = src1.read(3).astype(int)     # Cloud mask band
            unc = src1.read(2).astype(float) * 0.01  # Uncertainty band
            blue_band=src2.read(1)
            green_band=src2.read(2)
            red_band=src2.read(3)
            nir_band=src2.read(4)
            swir1_band=src2.read(5)
            swir2_band=src2.read(6)
            # Apply cloud mask
            mask = cloud_mask(cloud)
            red = red_band * 0.0000275 -0.2
            nir = nir_band * 0.0000275 -0.2
            blue= blue_band * 0.0000275 -0.2
            green= green_band * 0.0000275 -0.2
            swir1= swir1_band * 0.0000275 -0.2
            swir2=swir2_band * 0.0000275 -0.2

            ndvi = (nir - red) / (nir + red)
            nume= 0.356* blue + 0.130*red +0.373*nir +.085*swir1 +0.072*swir2 -0.0018
            albedo=nume/1.016
            ndbi= (swir1-nir)/(swir1+nir)
           
            LST_K = band10 * 0.00341802 + 149
            LST_C = LST_K - 273.15

            # Apply masking
            lst2 = np.copy(LST_C)
            lst2[(mask == 1)] = np.nan  
            nan_count=np.isnan(lst2).sum()
            if nan_count>= 0.4* lst2.shape[0]*lst2.shape[1]:
                return (np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan)
            
            mean = np.nanmean(lst2)
            sd = np.nanstd(lst2)

            # Initialize UHI array
            uhi = np.zeros(lst2.shape)

            avg_urban_lst = 0
            avg_rural_lst = 0
            urban_count = 0
            rural_count = 0
            avg_urban_ndvi=0
            avg_urban_ndbi=0
            avg_urban_albedo=0
            avg_rural_ndvi=0
            avg_rural_ndbi=0
            avg_rural_albedo=0
            # Classify urban and rural pixels
            for i in range(lst2.shape[0]):
                for j in range(lst2.shape[1]):
                    if np.isnan(lst2[i][j]):
                        continue  

                    if lst2[i][j] > mean + 0.5 * sd:
                        uhi[i][j] = 1  # Urban
                        avg_urban_lst += lst2[i][j]
                        urban_count += 1
                        avg_urban_ndvi+= ndvi[i][j]
                        avg_urban_albedo+=albedo[i][j]
                        avg_urban_ndbi+=ndbi[i][j]
                    else:
                        uhi[i][j] = -1  # Rural
                        avg_rural_lst += lst2[i][j]
                        rural_count += 1
                        avg_rural_ndvi+= ndvi[i][j]
                        avg_rural_albedo+=albedo[i][j]
                        avg_rural_ndbi+=ndbi[i][j]


            if urban_count > 0:
                avg_urban_lst /= urban_count
                avg_urban_ndvi/=urban_count
                avg_urban_ndbi/=urban_count
                avg_urban_albedo/=urban_count
            else:
                avg_urban_lst = np.nan
                avg_urban_ndvi = np.nan
                avg_urban_ndbi = np.nan
                avg_urban_albedo = np.nan

            if rural_count > 0:
                avg_rural_lst /= rural_count
                avg_rural_ndvi /= rural_count
                avg_rural_ndbi /= rural_count
                avg_rural_albedo /= rural_count
            else:
                avg_rural_lst = np.nan
                avg_rural_ndvi = np.nan
                avg_rural_ndbi = np.nan
                avg_rural_albedo = np.nan 

            if not np.isnan(avg_urban_lst) and not np.isnan(avg_rural_lst):
                uhi_intensity = avg_urban_lst - avg_rural_lst
                uhi_ratio = uhi_intensity / avg_rural_lst if avg_rural_lst != 0 else np.nan
            else:
                uhi_intensity = np.nan
                uhi_ratio = np.nan

            return (uhi_intensity, uhi_ratio,avg_urban_ndvi,avg_urban_ndbi,avg_urban_albedo,avg_rural_ndvi,avg_rural_ndbi,avg_rural_albedo)
    else:
        return (np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan)

