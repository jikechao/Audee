import tensorflow as tf
import argparse

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Pass the filename as an argument
parser = argparse.ArgumentParser()
parser.add_argument("--frozen_model_filename", default="D:\\server-backup-137\data\\tf_model\\lenet5-mnist_origin.pb", type=str,
                    help="Pb model file to import")
args = parser.parse_args()

# We load the protobuf file from the disk and parse it to retrieve the
# unserialized graph_def
with tf.io.gfile.GFile(args.frozen_model_filename, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

    # saver=tf.train.Saver()
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
        saver = tf.train.Saver()
        sess = tf.Session(graph=graph)
        # init_op = tf.global_variables_initializer()

        save_path = saver.save(sess, "./lenet5.ckpt")