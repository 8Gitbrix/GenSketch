import numpy as np
import os
import io
import json
import argparse
import tensorflow as tf
from six.moves import range
import svgwrite
from cairosvg import svg2png
from PIL import Image
import model as sketch_rnn_model
import utils
from sketch_pix2seq_train import load_dataset, reset_graph, load_checkpoint


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def draw_strokes(data, svg_filename, factor=0.2, padding=50):
    """
    little function that displays vector images and saves them to .svg
    :param data:
    :param factor:
    :param svg_filename:
    :param padding:
    :return:
    """
    min_x, max_x, min_y, max_y = utils.get_bounds(data, factor)
    dims = (padding + max_x - min_x, padding + max_y - min_y)
    dwg = svgwrite.Drawing(size=dims)
    dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))
    lift_pen = 1
    abs_x = int(padding / 2) - min_x
    abs_y = int(padding / 2) - min_y
    p = "M%s, %s " % (abs_x, abs_y)
    # use lowcase for relative position
    command = "m"

    for i in range(len(data)):
        if lift_pen == 1:
            command = "m"
        elif command != "l":
            command = "l"
        else:
            command = ""
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        lift_pen = data[i, 2]
        p += command + str(x) + ", " + str(y) + " "
    the_color = "black"
    stroke_width = 1

    dwg.add(dwg.path(p).stroke(the_color, stroke_width).fill("none"))
    # dwg.save()

    svg_code = dwg.tostring()
    img = svg2png(bytestring=svg_code)
    image = Image.open(io.BytesIO(img))
    image = image.resize((28,28))
    # image = image.convert('1')
    aarr = np.asarray(image)
    # aarr = np.reshape(aarr, (28*28))
    # np.save('array', aarr)
    # image.save(svg_filename + '.png')
    return dims, dwg.tostring(), aarr


def load_env_compatible(data_dir, model_dir):
    """Loads environment for inference mode, used in jupyter notebook."""
    # modified https://github.com/tensorflow/magenta/blob/master/magenta/models/sketch_rnn/sketch_rnn_train.py
    # to work with depreciated tf.HParams functionality
    model_params = sketch_rnn_model.get_default_hparams()
    with tf.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
        data = json.load(f)
    fix_list = ['conditional', 'is_training', 'use_input_dropout', 'use_output_dropout', 'use_recurrent_dropout']
    for fix in fix_list:
        data[fix] = (data[fix] == 1)
    model_params.parse_json(json.dumps(data))

    return load_dataset(data_dir, model_params, inference_mode=True)


def load_model_compatible(model_dir):
    """Loads model for inference mode, used in jupyter notebook."""
    # modified https://github.com/tensorflow/magenta/blob/master/magenta/models/sketch_rnn/sketch_rnn_train.py
    # to work with depreciated tf.HParams functionality
    model_params = sketch_rnn_model.get_default_hparams()
    with tf.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
        data = json.load(f)
    fix_list = ['conditional', 'is_training', 'use_input_dropout', 'use_output_dropout', 'use_recurrent_dropout']
    for fix in fix_list:
        data[fix] = (data[fix] == 1)
    model_params.parse_json(json.dumps(data))

    model_params.batch_size = 1  # only sample one at a time
    eval_model_params = sketch_rnn_model.copy_hparams(model_params)
    eval_model_params.use_input_dropout = 0
    eval_model_params.use_recurrent_dropout = 0
    eval_model_params.use_output_dropout = 0
    eval_model_params.is_training = 0
    sample_model_params = sketch_rnn_model.copy_hparams(eval_model_params)
    sample_model_params.max_seq_len = 1  # sample one point at a time
    return [model_params, eval_model_params, sample_model_params]


def encode(input_images, session, model):
    unused_strokes = np.zeros(shape=[model.hps.batch_size, model.hps.max_seq_len + 1, 5], dtype=np.float32)
    return session.run(model.batch_z,
                       feed_dict={model.input_data: unused_strokes,
                                  model.input_image: input_images})[0]


def decode(session, sample_model, max_seq_len, z_input=None, temperature=0.1):
    z = None
    if z_input is not None:
        z = [z_input]

    sample_strokes, m = sketch_rnn_model.sample(session, sample_model,
                                                seq_len=max_seq_len, temperature=temperature, z=z)
    strokes = utils.to_normal_strokes(sample_strokes)  # sample_strokes in stroke-5 format, strokes in stroke-3 format
    return strokes


def sampling_conditional(data_dir, sampling_dir, model_dir):
    [train_set, valid_set, test_set, hps_model, eval_hps_model, sample_hps_model] = \
        load_env_compatible(data_dir, model_dir)

    # construct the sketch-rnn model here:
    reset_graph()
    model = sketch_rnn_model.Model(hps_model)
    eval_model = sketch_rnn_model.Model(eval_hps_model, reuse=True)
    sampling_model = sketch_rnn_model.Model(sample_hps_model, reuse=True)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # loads the weights from checkpoint into our model
    load_checkpoint(sess, model_dir)

    iters = 10
    outputs = 4

    input_arr = np.zeros((iters, 28,28,3))
    group_out_arr = np.zeros((iters, outputs, 28,28,3))

    #Generate output to be displayed
    for its in range(iters):

        stroke, rand_idx, image = test_set.random_sample()
        sub_sampling_dir = os.path.join(sampling_dir, str(rand_idx))

        print("Random Image : {}/{}".format( its, iters))
        _, _, in_arr = draw_strokes(stroke, os.path.join(sub_sampling_dir, 'sample_gt.svg'))

        z = encode(image, sess, eval_model)

        for i in range(outputs):
            strokes_out = decode(sess, sampling_model, eval_model.hps.max_seq_len, z,
                                 temperature=0.1)  # in stroke-3 format
            _, _, grp_out_arr = draw_strokes(strokes_out, os.path.join(sub_sampling_dir, 'sample_pred_cond.svg'))
            group_out_arr[its, i] = grp_out_arr

        input_arr[its] = in_arr
    np.save('test_images.npy', input_arr)
    np.save('group_generated.npy', group_out_arr)

    #Generate output to be stored
    iters = 100
    generated = np.zeros((iters, 28, 28, 3))
    for its in range(iters):

        stroke, rand_idx, image = test_set.random_sample()
        sub_sampling_dir = os.path.join(sampling_dir, str(rand_idx))
        print("Random image : {}/{}".format( its, iters))

        z = encode(image, sess, eval_model)

        strokes_out = decode(sess, sampling_model, eval_model.hps.max_seq_len, z, temperature=0.1)
        _, _, out_arr = draw_strokes(strokes_out, os.path.join(sub_sampling_dir, 'sample_pred_cond.svg'))

        generated[its] = out_arr
    np.save('generated.npy', generated)


def main(**kwargs):
    data_dir_ = kwargs['data_dir']
    model_dir_ = kwargs['model_dir']
    sampling_dir_ = kwargs['sampling_dir']

    sampling_conditional(data_dir_, sampling_dir_, model_dir_)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-db', type=str, default='datasets', help="set the data base dir")
    parser.add_argument('--model_dir', '-md', type=str, default='outputs/snapshot', help="set the trained model dir")
    parser.add_argument('--sampling_dir', '-sd', type=str, default='outputs/sampling', help="set the results dir")
    args = parser.parse_args()

    run_params = {
        "data_dir": args.data_dir,
        "model_dir": args.model_dir,
        "sampling_dir": args.sampling_dir,
    }

    main(**run_params)
