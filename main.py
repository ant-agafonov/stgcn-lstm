import configparser
import argparse
import os.path
import shutil

from models.recurrent.LSTM import LSTMModel
from models.recurrent.GRU import GRUModel
from models.gcn.STGCNHybrid import STGCNHybridModel

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="configuration file path", required=True)
    parser.add_argument("--force", type=str, default=False, help="remove params dir", required=False)
    args = parser.parse_args()

    # read configuration
    config = configparser.ConfigParser()
    print('Read configuration file: %s' % args.config)
    config.read(args.config)

    training_config = config['Training']
    retrain = training_config['retrain'] == 'True'  # train or evaluate the model
    model_name = training_config['model_name']
    params_dir = training_config['params_dir']
    trained_model_name = params_dir + model_name + ".h5"
    if not retrain and not os.path.exists(trained_model_name):
        # training is not needed but the trained model is not found
        raise SystemExit('The saved trained model "' + trained_model_name + '" is not found')

    if os.path.exists(params_dir) and not args.force:
        raise SystemExit("Params folder exists! Select a new params path please!")
    if retrain:
        # clear params dir
        if os.path.exists(params_dir):
            shutil.rmtree(params_dir)
        os.makedirs(params_dir)
        # save the config file into the params dir
        shutil.copyfile(args.config, params_dir + os.path.basename(args.config))

    # create the model
    model_name = training_config['model_name']
    model = None
    if model_name == 'LSTM':
        model = LSTMModel(config)
    elif model_name == 'GRU':
        model = GRUModel(config)
    elif model_name == 'STGCNHybrid':
        model = STGCNHybridModel(config)

    # train or load the trained model
    if retrain:
        # train the model
        model.train(model_name)
    else:
        # load the trained model
        model.load(model_name)

    # evaluate the model
    model.evaluate()
    print("done")


if __name__ == "__main__":
    main()
