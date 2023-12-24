import tensorflowjs as tfjs

tfjs.converters.convert_tf_saved_model(
    saved_model_dir='models/my-model-27',
    output_dir='models/my-model-27-convert'
)