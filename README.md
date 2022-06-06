# gesture-recognition
ML based service for gesture recognition. Task: https://www.sciencedirect.com/science/article/abs/pii/S1574119209000674?via%3Dihub

Project divided into the following components:
- data: folder that contains raw, extracted, and clean data. It acts as data lake
- DataPipeline: contains pipelines to collect and extract data and save it into the data folder
- EDA: Contains notebooks for prototyping and analysis
- local_models_registry: Local folder to save the models in
- MLPipeline: pipeline for ML modeling (data preparation, training, validation)
- utils: for common utilities to be used across components