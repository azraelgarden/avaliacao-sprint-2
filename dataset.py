from openml.datasets import dataset


class Datasets:
    
    def __init__(self, id):
        self._id = id
    
    def __request(self):
        import openml

        dataset = openml.datasets.get_dataset(self._id)

        return dataset

    def get(self):
        dataset = self.__request()
        x, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)

        return x, y, categorical_indicator, attribute_names