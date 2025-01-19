import os, glob, datetime, time, cartopy, gc
import matplotlib as mpl, numpy as np, matplotlib.pyplot as plt, pandas as pd
from netCDF4 import Dataset
import ee
import rasterio
import geemap
from rasterio.merge import merge   

ETcolors = ["#f6e8c3", "#d8b365", "#99974a", "#53792d", "#6bdfd2", "#1839c5"]
conf0 = {
    'ALBEDO':{
    'NAME':'ALBEDO',
    'LONGNAME':'Shortwave BSA Albedo',
    'DATASET':'MODIS/061/MCD43A3',
    'SOURCE':'MODIS',
    'LS_BAND':['Albedo_BSA_shortwave'],
    'LS_BAND_QC':[],
    'FREQUENCY':'daily',
    'OUTPUT_SCALE':5000,
    'UNIT':'None',
    'VALI_RANGE':[0, 1000],
    'PLOT_RANGE':[0., 1.],
    'ANM_RANGE':[-0.25, 0.25],
    'VARCOLORMAP':'gist_ncar',
    'SCALE_FACTOR':0.001,
    'OFFSET_FACTOR':0.,
    'OUTPUT_SCALE_FACTOR':0.001,
    'OUTPUT_OFFSET_FACTOR':0.},
    'ALBEDO_sf':{
    'NAME':'ALBEDO_sf',
    'LONGNAME':'Snow-free Shortwave BSA Albedo',
    'DATASET':'MODIS/061/MCD43A3',
    'SOURCE':'MODIS',
    'LS_BAND':['Albedo_BSA_shortwave'],
    'LS_BAND_QC':[],
    'FREQUENCY':'daily',
    'OUTPUT_SCALE':5000,
    'UNIT':'None',
    'VALI_RANGE':[0, 1000],
    'PLOT_RANGE':[0., 1.],
    'ANM_RANGE':[-0.08, 0.08],
    'VARCOLORMAP':'gist_ncar',
    'SCALE_FACTOR':0.001,
    'OFFSET_FACTOR':0.,
    'OUTPUT_SCALE_FACTOR':0.001,
    'OUTPUT_OFFSET_FACTOR':0.},
    'AIRT':{
    'NAME':'AIRT',
    'LONGNAME':'2m Air Temperature',
    'DATASET':'ECMWF/ERA5_LAND/MONTHLY_AGGR',
    #'DATASET':'ECMWF/ERA5_LAND/MONTHLY_BY_HOUR',
    'SOURCE':'ECMWF',
    'LS_BAND':['temperature_2m'],
    'LS_BAND_QC':[],
    'FREQUENCY':'monthly',
    #'FREQUENCY':'daily',
    'OUTPUT_SCALE':5000,
    'UNIT':'K',
    'VALI_RANGE':[10000, 18000],
    'PLOT_RANGE':[213., 343.],
    'ANM_RANGE':[-15., 15.],
    'VARCOLORMAP':'jet',
    'SCALE_FACTOR':1.,
    'OFFSET_FACTOR':0.,
    'OUTPUT_SCALE_FACTOR':0.02,
    'OUTPUT_OFFSET_FACTOR':0.},
    'SM':{
    'NAME':'SM',
    'LONGNAME':'Soil Moisture',
    'DATASET':'NASA/SMAP/SPL4SMGP/007',
    'SOURCE':'SMAP',
    'LS_BAND':['sm_surface'],
    'LS_BAND_QC':[],
    'FREQUENCY':'3-hourly',
    'OUTPUT_SCALE':5000,
    'UNIT':'None',
    'VALI_RANGE':[0, 9000],
    'PLOT_RANGE':[0, 0.9],
    'ANM_RANGE':[-0.15, 0.15],
    'VARCOLORMAP':'RdYlGn',
    'SCALE_FACTOR':1.,
    'OFFSET_FACTOR':0.,
    'OUTPUT_SCALE_FACTOR':0.0001,
    'OUTPUT_OFFSET_FACTOR':0.},
    'NDVI':{
    'NAME':'NDVI',
    'LONGNAME':'NDVI',
    'DATASET':'MODIS/061/MYD13A2',
    'SOURCE':'MODIS',
    'LS_BAND':['NDVI'],
    'LS_BAND_QC':['DetailedQA'], # Only pixels with bit 2-5 < 13 are useful
    'LS_QC_BITS':[[2, 4, 13]], # [bit_start, bit_len, bit_threshold]
    'FREQUENCY':'16-daily',
    'OUTPUT_SCALE':5000,
    'UNIT':'None',
    'VALI_RANGE':[-2000, 10000],
    'PLOT_RANGE':[-0.2, 1.],
    'ANM_RANGE':[-0.25, 0.25],
    'VARCOLORMAP':'YlGn',
    'SCALE_FACTOR':0.0001,
    'OFFSET_FACTOR':0.,
    'OUTPUT_SCALE_FACTOR':0.001,
    'OUTPUT_OFFSET_FACTOR':0.},
    'EVI':{
    'NAME':'EVI',
    'LONGNAME':'EVI',
    'DATASET':'MODIS/061/MYD13A2',
    'SOURCE':'MODIS',
    'LS_BAND':['EVI'],
    'LS_BAND_QC':['DetailedQA'], # Only pixels with bit 2-5 < 13 are useful
    'LS_QC_BITS':[[2, 4, 13]], # [bit_start, bit_len, bit_threshold]
    'FREQUENCY':'16-daily',
    'OUTPUT_SCALE':5000,
    'UNIT':'None',
    'VALI_RANGE':[-2000, 10000],
    'PLOT_RANGE':[-0.2, 1.],
    'ANM_RANGE':[-0.25, 0.25],
    'VARCOLORMAP':'YlGn',
    'SCALE_FACTOR':0.0001,
    'OFFSET_FACTOR':0.,
    'OUTPUT_SCALE_FACTOR':0.001,
    'OUTPUT_OFFSET_FACTOR':0.},
    'LST_day':{
    'NAME':'LST_day',
    'LONGNAME':'Daytime Land Surface Temperature',
    'DATASET':'MODIS/061/MYD11A1',
    'SOURCE':'MODIS',
    'LS_BAND':['LST_Day_1km'],
    'LS_BAND_QC':[],
    'FREQUENCY':'daily',
    'OUTPUT_SCALE':5000,
    'UNIT':'K',
    'VALI_RANGE':[10000, 18000],
    'PLOT_RANGE':[213., 343.],
    'ANM_RANGE':[-15., 15.],
    'VARCOLORMAP':'jet',
    'SCALE_FACTOR':0.02,
    'OFFSET_FACTOR':0.,
    'OUTPUT_SCALE_FACTOR':0.02,
    'OUTPUT_OFFSET_FACTOR':0.},
    'LST_night':{
    'NAME':'LST_night',
    'LONGNAME':'Nighttime Land Surface Temperature',
    'DATASET':'MODIS/061/MYD11A1',
    'SOURCE':'MODIS',
    'LS_BAND':['LST_Night_1km'],
    'LS_BAND_QC':[],
    'FREQUENCY':'daily',
    'OUTPUT_SCALE':5000,
    'UNIT':'K',
    'VALI_RANGE':[10000, 18000],
    'PLOT_RANGE':[213., 343.],
    'ANM_RANGE':[-15., 15.],
    'VARCOLORMAP':'jet',
    'SCALE_FACTOR':0.02,
    'OFFSET_FACTOR':0.,
    'OUTPUT_SCALE_FACTOR':0.02,
    'OUTPUT_OFFSET_FACTOR':0.},
    'CLEAR_SKY':{   # the Clear_day_cov in LST
    'NAME':'CLEAR_SKY',
    'LONGNAME':'Clear Sky',
    'DATASET':'MODIS/061/MYD11A1',
    'SOURCE':'MODIS',
    'LS_BAND':['Clear_day_cov'],
    'LS_BAND_QC':[],
    'FREQUENCY':'daily',
    'OUTPUT_SCALE':5000,
    'UNIT':'None',
    'VALI_RANGE':[1, 2000],
    'PLOT_RANGE':[0., 1.],
    'ANM_RANGE':[-1, 1.], # to be determined
    'VARCOLORMAP':'jet', # to be determined
    'SCALE_FACTOR':0.0005,
    'OFFSET_FACTOR':0.,
    'OUTPUT_SCALE_FACTOR':0.0005,
    'OUTPUT_OFFSET_FACTOR':0.},
    'ET':{
    'NAME':'ET',
    'LONGNAME':'Evapotranspiration',
    'DATASET':'MODIS/061/MOD16A2',
    'SOURCE':'MODIS',
    'LS_BAND':['ET'],
    'LS_BAND_QC':[],
    'FREQUENCY':'8-daily',
    'OUTPUT_SCALE':5000,
    'UNIT':'kg/m^2/8day',
    'VALI_RANGE':[0, 700],
    'PLOT_RANGE':[0., 65.],
    'ANM_RANGE':[-10, 10],
    'VARCOLORMAP':mpl.colors.LinearSegmentedColormap.from_list("ET", ETcolors),
    'SCALE_FACTOR':0.1,
    'OFFSET_FACTOR':0.,
    'OUTPUT_SCALE_FACTOR':0.1,
    'OUTPUT_OFFSET_FACTOR':0.},
    'LAI':{
    'NAME':'LAI',
    'LONGNAME':'Leaf Area Index',
    'DATASET':'MODIS/061/MCD15A3H',
    'SOURCE':'MODIS',
    'LS_BAND':['Lai'],
    'LS_BAND_QC':[],
    'FREQUENCY':'4-daily',
    'OUTPUT_SCALE':5000,
    'UNIT':'None',
    'VALI_RANGE':[0, 100],
    'PLOT_RANGE':[0., 7.],
    'ANM_RANGE':[-1, 1],
    'VARCOLORMAP':'YlGn',
    'SCALE_FACTOR':0.1,
    'OFFSET_FACTOR':0.,
    'OUTPUT_SCALE_FACTOR':0.1,
    'OUTPUT_OFFSET_FACTOR':0.},
    'PRCP_GPM':{
    'NAME':'PRCP_GPM',
    'LONGNAME':'Precipitation',
    'DATASET':'NASA/GPM_L3/IMERG_V07',
    'SOURCE':'GPM',
    'LS_BAND':['precipitation'],
    'LS_BAND_QC':[],
    'FREQUENCY':'daily',
    'OUTPUT_SCALE':5000,
    'UNIT':'mm',
    'VALI_RANGE':[0, 10000],
    'PLOT_RANGE':[0., 70.],
    'ANM_RANGE':[-50,50],
    'VARCOLORMAP':'YlGnBu',
    'SCALE_FACTOR':1.,
    'OFFSET_FACTOR':0.,
    'OUTPUT_SCALE_FACTOR':1.,
    'OUTPUT_OFFSET_FACTOR':0.}
}

def plot_data(data, fname_png, title, ctitle='Temperature(°C)',   \
    drange=(213., 343.), cmap=None, norm=None, cb_extend=None,           \
    figsize=(10., 6.25), pos_map=[0.0, 0.13, 1.0, 0.8],                  \
    pos_cb=[0.2, 0.09, 0.6, 0.02], interp='none', extent=None,           \
    us_state=False, r_border=0.5):
    if cmap is None: cmap = plt.cm.jet
    ratio = figsize[0]/10.
    dproj = cartopy.crs.PlateCarree(central_longitude=0.)
    proj = cartopy.crs.PlateCarree(central_longitude=0.)
    if norm is None:
      norm = mpl.colors.Normalize(vmin=drange[0], vmax=drange[1])
    fig = plt.figure(figsize=figsize, facecolor='w')
    ax = plt.axes(pos_map, projection=proj, facecolor='w')
    if extent is None: extent = ax.get_extent()
    rr = data.shape[1] / (extent[1] - extent[0])
    ax.add_feature(cartopy.feature.BORDERS, linewidth=r_border*ratio)
    ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.25*ratio)
    if us_state:
      ax.add_feature(cartopy.feature.STATES, linewidth=0.25*ratio)
    fname_bg = '/content/drive/MyDrive/Pics/python/shaded.relief.0d05.jpg'
    image = plt.imread(fname_bg)
    ax.imshow(image, extent=[-180., 180., -90., 90.],transform=dproj)
    g1=ax.gridlines(crs=dproj, linewidth=1, color='black',alpha=0.5,     \
      linestyle='--')
    g1.xlocator=mpl.ticker.FixedLocator(range(-180, 181, 60))
    g1.ylocator = mpl.ticker.FixedLocator(range(-90, 91, 30))
    ax.imshow(data, cmap=cmap, norm=norm, transform=dproj, extent=extent, \
      interpolation=interp)
    plt.title(title, fontsize=20*ratio, color='black')
    ax3 = fig.add_axes(pos_cb, facecolor='k')
    cb = mpl.colorbar.ColorbarBase(ax3, cmap=cmap, norm=norm,            \
      orientation='horizontal', extend=cb_extend)
    cb.set_label(ctitle, fontsize=14*ratio, color='black')
    cb.ax.xaxis.set_tick_params(color='black', labelsize=12*ratio)
    cb.outline.set_edgecolor('black')
    plt.setp(plt.getp(cb.ax.axes, 'xticklabels'), color='black')
    plt.savefig(fname_png, facecolor='white')
    plt.clf()
    plt.close()
    del data, image

def plot_data2(data, fname_png, drange=(213., 343.), cmap=None,          \
    norm=None, figsize=None, pos_map=[0.0, 0.0, 1.0, 1.0], interp='none'):
    if cmap is None: cmap = plt.cm.jet
    if norm is None:
      norm = mpl.colors.Normalize(vmin=drange[0], vmax=drange[1])
    if figsize is None: figsize = (data.shape[1]/100., data.shape[0]/100.)
    dproj = cartopy.crs.PlateCarree(central_longitude=0.)
    proj = cartopy.crs.PlateCarree(central_longitude=0.)
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(pos_map, projection=proj)
    ax.axis('off')
    ax.imshow(data, cmap=cmap, norm=norm, transform=dproj,              \
      extent=(-180., 180., -90., 90.), interpolation=interp)
    plt.savefig(fname_png, transparent=True)
    plt.clf()
    plt.close()
    del data


#--- Functions ---
def maskQC(img):
    qc_var = ((2**(bitlen)-1) << bitstart)
    qc_thr = (thres << bitstart)
    return img.updateMask(img.select(cur_band_qc).bitwiseAnd(qc_var).lte(qc_thr))

def maskValidRange(img):
    mask1 = img.lte(VALI_RANGE[1])
    mask2 = img.gte(VALI_RANGE[0])
    return img.updateMask(mask1).updateMask(mask2)

def maskValidRange_Uplimit(img):
    mask0 = img.gt(VALI_RANGE[1]).multiply(1)
    mask1 = img.lte(VALI_RANGE[1])
    mask2 = img.gte(VALI_RANGE[0])
    return img.updateMask(mask1).updateMask(mask2).add(mask0)

def applyScaler(img):
    return img.multiply(SCALE_FACTOR).add(OFFSET_FACTOR)

def resampling5km(img):
    return img.reduceResolution(reducer = ee.Reducer.mean(), maxPixels = 144).reproject(crs = PROJ_5KM)

def resampling5km_PRCP(img):
    return img.reproject(crs = PROJ_5KM).reduceResolution(reducer = ee.Reducer.mean(), maxPixels = 144)

def resampling25km(img):
    return img.reduceResolution(reducer = ee.Reducer.mean(), maxPixels = 144).reproject(crs = 'EPSG:4326', scale=25000)

def sumHourly(dayOffset, imgs, ee_startDate, ee_endDate):
    start = ee_startDate.advance(dayOffset, 'days')
    end = start.advance(1, 'days')
    return imgs.filterDate(start, end).sum().set({'system:index':start.format('YYYY-MM-DD')})

def maskA3(img, cur_qc):
    return img.updateMask(img.select(cur_qc).eq(0))

def Match_mcd(img):
    # Create a collection with 1 image
    temp = ee.ImageCollection(ee.List([img]))
    # Apply join to collection 2
    join = simpleJoin.apply(mod2join, temp, join_TimeFilter)
    # Get resulting image
    i2 = ee.Image(join.first())
    return img.addBands(i2)

def maskSnowfree(image):
    mask2 = image.select('BRDF_Albedo_LocalSolarNoon').lt(85)
    mask3 = image.select('Snow_BRDF_Albedo').eq(0)
    return image.updateMask(mask3).updateMask(mask2)

def maskSnowincluded(image):
    mask2 = image.select('BRDF_Albedo_LocalSolarNoon').lt(85)
    return image.updateMask(mask2)

def get_collection_lastdate(mod):
    ll = mod.size().getInfo()
    ccc = ee.Image(mod.toList(ll).get(-1))
    ddd = ee.Date(ccc.get('system:time_start')).format('YYYY-MM-dd').getInfo()
    ddd = datetime.datetime.strptime(ddd, '%Y-%m-%d').date()
    return ddd

def determine_complete(date, datebeg, frequency):
    date1_nm = (datebeg+datetime.timedelta(31)).replace(day=1)
    ddays = (date1_nm-date).days
    dummy = frequency.split('-')
    if dummy[-1] == 'daily':
        if len(dummy) == 1:
            ddays0 = 1
        else:
            ddays0 = int(dummy[0])
    else:
        ddays0 = (date1_nm - datebeg).days
    result = ddays <= ddays0
    return result

ee.Authenticate()
ee.Initialize(project='ee-peng8479')
RESAMPLING = True
PROJ_5KM = ee.ImageCollection('Oxford/MAP/TCB_5km_Monthly').first().projection()
OUTPUT_CRS = 'EPSG:4326'

vvars = ['ALBEDO']
datebeg = datetime.date(2024, 12, 1)
dateend = (datetime.date.today().replace(day=1) - datetime.timedelta(1)).replace(day=1)
ndays = (dateend - datebeg).days + 1
dir_root0 = '/Users/omtailor/GEE'
dir_out0 = os.path.join(dir_root0, 'geotiff')
os.makedirs(dir_out0, exist_ok=True)
os.chdir(dir_out0)

for vvar in vvars:
    conf = conf0[vvar]
    for dummy in conf:
        exec('{0} = conf["{0}"]'.format(dummy))
    cres = '{}km'.format(str(round(OUTPUT_SCALE/1000)))
    dir_root = os.path.join(dir_root0, 'monthly_{}'.format(cres))
    dir_data = os.path.join(dir_root, vvar, 'Data')
    for dd in range(ndays):
        date = datebeg + datetime.timedelta(dd)
        if date.day != 1:
            continue
        fname = os.path.join(dir_data, '{0}/{1}_{2}_{3}.nc'.format(date.strftime('%Y'), vvar, date.strftime('%Y%m'), cres))
        if os.path.exists(fname):
            continue
        cur_dateStart = date.strftime('%Y-%m-%d')
        datetmp = (date + datetime.timedelta(31)).replace(day=1)
        cur_dateEnd = datetmp.strftime('%Y-%m-%d')
        cur_month = date.strftime('%m')
        cur_year = date.strftime('%Y')
        dir_out = os.path.join(dir_out0, cur_year)
        os.makedirs(dir_out, exist_ok=True)
        print('\nStart processing '+vvar+': '+cur_dateStart+' and '+cur_dateEnd)
        mod = ee.ImageCollection(DATASET).filterDate(cur_dateStart, cur_dateEnd).select(LS_BAND+LS_BAND_QC)
        print(mod.size().getInfo())
        if mod.size().getInfo() == 0:
            print('\tNo data in this temporal range ')
            continue
        date_last = get_collection_lastdate(mod)
        if not determine_complete(date_last, date, FREQUENCY):
            continue
        for cur_band_idx in range(len(LS_BAND)):
            cur_band = LS_BAND[cur_band_idx]
            cur_mod = mod.select(cur_band)
            print('\ttarget band: '+cur_band+'..')
            if vvar == 'ALBEDO':
                bands_other = ['Snow_BRDF_Albedo','BRDF_Albedo_LocalSolarNoon']
                mod2 = ee.ImageCollection("MODIS/061/MCD43A2").filterDate(cur_dateStart, cur_dateEnd).select(bands_other)
                join_TimeFilter = ee.Filter.equals(leftField='system:index', rightField='system:index')
                simpleJoin = ee.Join.simple()
                mod1join = ee.ImageCollection(simpleJoin.apply(cur_mod, mod2, join_TimeFilter))
                mod2join = ee.ImageCollection(simpleJoin.apply(mod2, cur_mod, join_TimeFilter))
                mod = mod1join.map(Match_mcd)
                mod_si = mod.map(maskSnowincluded).select(cur_band)
                mod_sf = mod.map(maskSnowfree).select(cur_band)
                mods = {'ALBEDO':mod_si, 'ALBEDO_sf':mod_sf}
            else:
                if vvar == 'PRCP_GPM':
                    ee_startDate = ee.Date(cur_dateStart)
                    ee_endDate = ee.Date(cur_dateEnd)
                    numberOfDays = ee_endDate.difference(ee_startDate, 'days')
                    cur_mod = ee.ImageCollection(
                        ee.List.sequence(0, numberOfDays.subtract(1)).map(
                            lambda dayOffset: sumHourly(dayOffset, cur_mod, ee_startDate, ee_endDate)
                        )
                    )
                mods = {vvar:cur_mod}
            for vvar2 in mods.keys():
                if len(LS_BAND_QC) > 0:
                    cur_band_qc = LS_BAND_QC[cur_band_idx]
                    cur_band = LS_BAND[cur_band_idx]
                    bitstart, bitlen, thres = LS_QC_BITS[cur_band_idx]
                    modtmp = mod.select([cur_band, cur_band_qc])
                    cur_mod = modtmp.map(maskQC).select(cur_band)
                else:
                    cur_mod = mods[vvar2]
                if vvar == 'CLEAR_SKY':
                    cur_mod = cur_mod.map(maskValidRange_Uplimit)
                else:
                    cur_mod = cur_mod.map(maskValidRange)
                if RESAMPLING:
                    if vvar2 == 'PRCP_GPM' or vvar2 == 'PRCP_ERA':
                        cur_mod = cur_mod.map(resampling5km_PRCP)
                    else:
                        if OUTPUT_SCALE == 25000:
                            cur_mod = cur_mod.map(resampling25km)
                        else:
                            cur_mod = cur_mod.map(resampling5km)
                dummy = FREQUENCY.split('-')
                res_mean = cur_mod.reduce(ee.Reducer.mean())
                if dummy[-1] == 'monthly':
                    cc = ['mean']
                else:
                    cc = ['mean', 'std', 'num']
                    res_std = cur_mod.reduce(ee.Reducer.stdDev())
                    res_num = cur_mod.reduce(ee.Reducer.count())

                for ccase in cc:
                    description = '{0}_{1}{2}_{3}_{4}'.format(vvar2, cur_year, cur_month, ccase, cres)
                    exec('data = res_{}'.format(ccase))

                    # --------------------------------------
                    # Use geemap to download multiple tiles
                    # --------------------------------------
                    roi = ee.Geometry.BBox(-180, -90, 180, 90)
                    features = ee.FeatureCollection([ee.Feature(roi)])
                    
                    # This will split the global image into multiple tiles,
                    # each with dimensions up to max_tile_dim=100.
                    geemap.download_ee_image_tiles(
                        image=data,
                        features=features,
                        out_dir=dir_out,
                        prefix=description,
                        crs=OUTPUT_CRS,
                        scale=OUTPUT_SCALE,
                        resampling='near',
                        overwrite=True,
                        max_tile_dim=100, 
                        max_tile_size=16
                    )

                    # ---------------------------------------
                    # Mosaic the tiles back to a single file 
                    # ---------------------------------------
                    search_pattern = os.path.join(dir_out, f"{description}*.tif")
                    tile_paths = glob.glob(search_pattern)
                    if len(tile_paths) > 1:
                        src_files_to_mosaic = []
                        for fp in tile_paths:
                            src = rasterio.open(fp)
                            src_files_to_mosaic.append(src)
                        mosaic, out_trans = merge(src_files_to_mosaic)
                        out_meta = src_files_to_mosaic[0].meta.copy()
                        out_meta.update({
                            "driver": "GTiff",
                            "height": mosaic.shape[1],
                            "width": mosaic.shape[2],
                            "transform": out_trans
                        })
                        merged_path = os.path.join(dir_out, f"{description}_merged.tif")
                        with rasterio.open(merged_path, "w", **out_meta) as dest:
                            dest.write(mosaic)
                        print(f"Merged {len(tile_paths)} tiles into {merged_path}")

fillvalue = -32768
vvars = ['ALBEDO']
datebeg = datetime.date(2024, 12, 1)
dateend = (datetime.date.today().replace(day=1) - datetime.timedelta(1)).replace(day=1)
ndays = (dateend - datebeg).days + 1
dir_root0 = '/Users/omtailor/GEE'
dir_in = os.path.join(dir_root0, 'geotiff')

for vvar in vvars:
    conf = conf0[vvar]
    for dummy in conf:
        exec('{0} = conf["{0}"]'.format(dummy))
    cres = '{}km'.format(str(round(OUTPUT_SCALE/1000)))
    dir_root = os.path.join(dir_root0, 'monthly_{}'.format(cres))
    dir_data = os.path.join(dir_root, vvar, 'Data')
    dummy = FREQUENCY.split('-')
    is_monthly = dummy[-1] == 'monthly'
    
    for dd in range(ndays):
        date = datebeg + datetime.timedelta(dd)
        if date.day != 1:
            continue
        fname = os.path.join(dir_data, '{0}/{1}_{2}_{3}.nc'.format(date.strftime('%Y'), vvar, date.strftime('%Y%m'), cres))
        if os.path.exists(fname):
            continue
        fname_mean = os.path.join(dir_in, '{0}/{1}_{2}_mean_{3}1.tif'.format(date.strftime('%Y'), vvar, date.strftime('%Y%m'), cres))
        print(f"{fname_mean=}")
        if not os.path.exists(fname_mean):
            continue
        fname_num = os.path.join(dir_in, '{0}/{1}_{2}_num_{3}1.tif'.format(date.strftime('%Y'), vvar, date.strftime('%Y%m'), cres))
        if not is_monthly and not os.path.exists(fname_num):
            continue
        fname_std = os.path.join(dir_in, '{0}/{1}_{2}_std_{3}1.tif'.format(date.strftime('%Y'), vvar, date.strftime('%Y%m'), cres))
        if not is_monthly and not os.path.exists(fname_std):
            continue
        
        # read data
        with rasterio.open(fname_mean) as tif_reader1:
            cur_mean = tif_reader1.read(1) * conf['SCALE_FACTOR'] + conf['OFFSET_FACTOR']

        if is_monthly:
            ccs = ['mean']
        else:
            ccs = ['mean', 'std', 'num']
            with rasterio.open(fname_std) as tif_reader2:
                cur_std = tif_reader2.read(1) * conf['SCALE_FACTOR'] + conf['OFFSET_FACTOR']
            with rasterio.open(fname_num) as tif_reader3:
                cur_num = tif_reader3.read(1)

        os.makedirs(os.path.dirname(fname), exist_ok=True)
        nlats, nlons = cur_mean.shape
        f2 = Dataset(fname, 'w', format='NETCDF4')
        f2.createDimension('y', nlats)
        f2.createDimension('x', nlons)
        f2.description = 'The monthly data was derived from GEE {}'.format(conf['DATASET'])
        for cc in ccs:
            exec('tmp = cur_{}'.format(cc))
            d1 = f2.createVariable(cc, 'i2', ('y', 'x'), zlib=True, complevel=3, fill_value=fillvalue)
            if cc != 'num':
                tmp = (tmp - conf['OUTPUT_OFFSET_FACTOR']) / conf['OUTPUT_SCALE_FACTOR']
                tmp = tmp.round()
                tmp[np.isnan(tmp)] = fillvalue
                tmp[np.isinf(tmp)] = fillvalue
                tmp = np.clip(tmp, -32768, 32767)
            d1[:] = tmp.astype(int)
            if cc == 'num':
                continue
            d1.scale_factor = conf['OUTPUT_SCALE_FACTOR']
            d1.add_offset = conf['OUTPUT_OFFSET_FACTOR']
            if cc == 'mean':
                d1.valid_range = conf['VALI_RANGE']
            else:
                am = (0. - conf['OUTPUT_OFFSET_FACTOR']) / conf['OUTPUT_SCALE_FACTOR']
                am = np.floor(am).astype(np.int16)
                d1.valid_range = [am, conf['VALI_RANGE'][1]]
            d1.units = conf['UNIT']
        f2.close()
