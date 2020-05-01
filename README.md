# fastai implementation of Single Image Super-Resolution
> How to use fastai for super resolution.


fastai is a very easy-to-use Deep Learning library.
I will explain how to make Databunch when performing Super-Resolution using fastai.

## Abstract

* How to create databunch for Super-Resolution.
* Implementation of Super-Resolution.

## How to create the databunch

### after open image

```python
def after_open_image(img:PIL.Image, size, scale:int=1, sizeup:bool=False, crop:bool=True, luminance:bool=False)->PIL.Image:
    """ after_open function of ImageImageList """
    w, h = img.size
    if scale > 1: img = lr_image(img, scale, sizeup=sizeup)
    if crop: img = crop_center_image(img, size)
    if luminance: img = split_luminance(img)
    return img
```

### x data

```python
src = ImageImageList.from_folder(data_path, convert_mode=convert_mode,
                                 after_open=partial(after_open_image, size=in_size, scale=scale, sizeup=sizeup, luminance=luminance))
```

### y data

```python
src = src.label_from_func((lambda x: x), label_cls=ImageImageList, convert_mode=convert_mode,
                           after_open=partial(after_open_image, size=out_size, luminance=luminance))
```

### transformers

```python
def get_sr_transforms(size, max_lighting:float=0.2, p_lighting:float=0.75, xtra_tfms:Optional[Collection[Transform]]=None)->Collection[Transform]:
    """ trainsorms for super-resolution """
    res = [crop(size=size)]
    if max_lighting:
        res.append(brightness(change=(0.5*(1-max_lighting), 0.5*(1+max_lighting)), p=p_lighting))
        res.append(contrast(scale=(1-max_lighting, 1/(1-max_lighting)), p=p_lighting))
    #       train                   , valid
    return (res + listify(xtra_tfms), [crop(size=size)])
```

### x transformer

```python
data = src.transform(get_sr_transforms(size=in_size), tfm_y=True)
```

### y transformer

```python
data = data.transform_y(get_sr_transforms(size=out_size, max_lighting=0))
```

### normalize

```python
if luminance: data.normalize(do_y=True)
else: data.normalize(imagenet_stats, do_y=True)
```

## Implementations of Super-Resolution.

* srcnn
* epscn
* srresnet
* unet(fastai)
* unet-floss(fastai)
