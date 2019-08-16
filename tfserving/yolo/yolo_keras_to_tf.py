import argparse
from keras import backend as K
from keras.models import load_model
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model.signature_def_utils import predict_signature_def
from tensorflow.python.saved_model import tag_constants


def export_tf_model(input_path, export_path):

    K.set_learning_phase(0)
    keras_model = load_model(input_path)

    builder = saved_model_builder.SavedModelBuilder(export_path)

    print(keras_model.input)
    print(keras_model.output)

    signature = predict_signature_def(inputs={"inputs": keras_model.input},
                                      outputs={"yolo_1": keras_model.output[0], "yolo_2": keras_model.output[1], "yolo_3": keras_model.output[2]})

    with K.get_session() as sess:
        builder.add_meta_graph_and_variables(sess=sess, tags=[tag_constants.SERVING],
                                             signature_def_map={"serving_default": signature})

    builder.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Full path to .h5 model', type=str)
    parser.add_argument(
        '--target', help='Full path to save model including the version (typically <path to the .h5 folder>/1/)', type=str)
    args = parser.parse_args()
    export_tf_model(args.model, args.target)