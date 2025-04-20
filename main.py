import re
from typing import Literal, Iterator

import numpy as np
import pandas as pd
import tensorflow as tf

# Turn down tensorflow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.compat.v1.enable_eager_execution()

EC_NUMBER_REGEX = r'(\d+).([\d\-n]+).([\d\-n]+).([\d\-n]+)'


class ProtNLMPredictor:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.__load_model()

    def __load_model(self):
        self.model = tf.saved_model.load(export_dir=self.model_path)
        self.model_infer = self.model.signatures['serving_default']

    @staticmethod
    def __prepare_sequence_query(sequence: str):
        return f"[protein_name_in_english] <extra_id_0> [sequence] {sequence}"

    @staticmethod
    def __decode_labeling(labeling):
        names = labeling['output_0'][0].numpy().tolist()
        scores = labeling['output_1'][0].numpy().tolist()
        beam_size = len(names)
        names = [names[beam_size - 1 - i].decode().replace('<extra_id_0> ', '') for i in range(beam_size)]

        for i, name in enumerate(names):
            if re.match(EC_NUMBER_REGEX, name):
                names[i] = 'EC:' + name

        scores = [np.exp(scores[beam_size - 1 - i]) for i in range(beam_size)]
        return names, scores

    def __predict_name_per_sequence(self, sequence: str):
        sequence_query = self.__prepare_sequence_query(sequence)
        labeling = self.model_infer(
            tf.constant([sequence_query], dtype=tf.string)
        )
        names, scores = self.__decode_labeling(labeling)
        top_name = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)[0][0]
        return top_name

    @staticmethod
    def __make_batch(sequences: list[str], batch_size: int = 256) -> Iterator[list[str]]:
        batch_count = len(sequences) // batch_size
        for i in range(batch_count + 1):
            yield sequences[i * batch_size: (i + 1) * batch_size]

    def predict(self, sequence: list[str], mode: Literal['single', 'batch'] = 'single'):
        if mode == 'single' and len(sequence) == 1:
            return [self.__predict_name_per_sequence(sequence[0])]
        else:
            name_predictions = []
            for batch in self.__make_batch(sequence):
                batch_prediction = tf.nest.map_structure(
                    tf.stop_gradient,
                    tf.map_fn(
                        self.__predict_name_per_sequence,
                        tf.constant(batch),
                        dtype=tf.string,
                        fn_output_signature=tf.string
                    ),
                )
                name_predictions.extend(batch_prediction.numpy().tolist())
                self.__clear_session()
            return name_predictions

    def __del__(self):
        del self.model
        del self.model_infer
        tf.keras.backend.clear_session()

    def __clear_session(self):
        tf.keras.backend.clear_session()


if __name__ == '__main__':
    predictor = ProtNLMPredictor(model_path='protnlm')
    sequence_df = pd.read_csv('data/biogrid_prepared/protein_data_biogrid.csv')
    predictor.predict(sequence_df['content'].to_list()[:5], mode='batch')
