## Using custom models

- Change path of model in specific config, for example [here](./anylabeling/configs/auto_labeling/m_retina_license_plate_mobilenetv3.yaml)

``` yaml
...
model_path: F:\Code\X-AnyLabeling\models\retina_license_plate_mobilenetv3.onnx
...
```


- Add custom config from [configs/auto_labeling/models_custom.yaml](./anylabeling/configs/auto_labeling/models_custom.yaml) to [configs/auto_labeling/models.yaml](./anylabeling/configs/auto_labeling/models.yaml)

- In [services/auto_labeling/model_manager.py](./anylabeling/services/auto_labeling/model_manager.py) 

Add custom model name to `CUSTOM_MODELS` of `ModelManager`

``` python
...
CUSTOM_MODELS = [
    ...,
    "retinanet"
]
...
```

Add custom model run scripts in `_load_model` function of `ModelManager`

``` python
...
elif model_config["type"] == "retinanet":
    from .m_retina_license_plate_mobilenetv3 import RetinaLicensePlateMobilenetV3

    try:
        model_config["model"] = RetinaLicensePlateMobilenetV3(
            model_config, on_message=self.new_model_status.emit
        )
        self.auto_segmentation_model_unselected.emit()
    except Exception as e:  # noqa
        self.new_model_status.emit(
            self.tr(
                "Error in loading model: {error_message}".format(
                    error_message=str(e)
                )
            )
        )
        print(
            "Error in loading model: {error_message}".format(
                error_message=str(e)
            )
        )
        return
... 
```

## Git update

``` bash
git stash
git pull
git stash pop
```