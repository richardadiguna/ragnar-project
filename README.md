# ragnar-project

This project is a convolutional neural network model with a little bit modification based on Bayar et al., 2016 to detect a manipulation/forgery on the image. This model can detect 6 classes: median filtering, pristine, gaussian blur, additive noise, resampling, jpeg compression.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development. See deployment for notes on how to deploy the project on a live system.

### Datasets
* The dataset for training the model is generated using ForgeryGenerator.py with source images in Dresden Raw Image dataset (jpeg format)
* The dataset will contains N numbers 128x128x1 images with N numbers of pristine patches and N numbers of median filtering, gaussian blur, additive gaussian, resampling, and jpeg compression patches

### Prerequisites

1. There are some dependecies you need to install into the environment. Just run requirement.sh in the terminal.

2. GPU must be installed into your hardware before loading the saved model.


### Installing Dependencies

Install all dependecies as easy as run requirement.sh file in this repo.

```bash
.requirement.sh
```

## Deployment

The deployment of this project can be done using docker and tf_serving:gpu-latest image.

## Versioning

Versioning in this project is done manully while generating saved_model.pb file

```bash
python3 serve/ServeModel.py -ver <VERSIOn> -sp <META_FILE_PATH> -sm <PB_FILE_PATH>
```

or

```bash
python3 serve/ServeModel.py -h
```

to see all details about the arguments while running ServeModel.py

## Client
To communicate with docker GRPC, client file use protobuf in tensorflow_serving_apis. You can find the code in clien/Client.py. All RPC communication can only use port 8500 for REST API 8501.

```python
def get_prediction_from_model(data):
    ...
    return
```

## Authors

* **Richard Adiguna** - *Initial work*

## License

This project is licensed under the KoinWorks License.

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
