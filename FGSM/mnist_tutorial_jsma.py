from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from keras import backend as K
import numpy as np
from operator import mul
import os

from six.moves import xrange
import math

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from utils_mnist import data_mnist
from utils_tf import model_train, model_eval

from attacks import jsma
from attacks_tf import jacobian_graph
from utils import other_classes, cnn_model, pair_visual, grid_visual



FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', '/Users/xiaowei/Repositories/DLV/FGSM/', 'Directory storing the saved model.')
flags.DEFINE_string('filename', 'mnist.ckpt', 'Filename to save model under.')
flags.DEFINE_boolean('viz_enabled', True, 'Enable sample visualization.')
flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_integer('nb_classes', 10, 'Number of classification classes')
flags.DEFINE_integer('img_rows', 28, 'Input row dimension')
flags.DEFINE_integer('img_cols', 28, 'Input column dimension')
flags.DEFINE_integer('nb_channels', 1, 'Nb of color channels in the input.')
flags.DEFINE_integer('nb_filters', 64, 'Number of convolutional filter to use')
flags.DEFINE_integer('nb_pool', 2, 'Size of pooling area for max pooling')
flags.DEFINE_integer('source_samples', 1, 'Nb of test set examples to attack')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')
flags.DEFINE_integer('starting_index', 4357, 'starting index of the image')
flags.DEFINE_integer('thetaValue', 1.0, 'theta Value')


def main(argv=None):
    """
    MNIST tutorial for the Jacobian-based saliency map approach (JSMA)
    :return:
    """

    os.environ['KERAS_BACKEND']='tensorflow'

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)
    
    fileName = "statistics/JAMA_dataCollection.txt"
    fileHandler = open(fileName, 'a')

    ###########################################################################
    # Define the dataset and model
    ###########################################################################

    # Image dimensions ordering should follow the Theano convention
    if K.image_dim_ordering() != 'tf':
        K.set_image_dim_ordering('tf')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' "
              "to 'th', temporarily setting to 'tf'")

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    K.set_session(sess)
    print("Created TensorFlow session and set Keras backend.")

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist()
    print("Loaded MNIST test data.")

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    # Define TF model graph
    model = cnn_model()
    predictions = model(x)
    print("Defined TensorFlow model graph.")
    
    
    #filename = "pic/%s.jpg"%(FLAGS.starting_index)
    #testImage = np.squeeze(X_test[(FLAGS.starting_index):(FLAGS.starting_index+1)][0])
    #print("%s--%s"%(str(np.amax(testImage)), str(np.amin(testImage))))
    #save(0,testImage,filename)
    

    ###########################################################################
    # Training the model using TensorFlow
    ###########################################################################

    # Train an MNIST model if it does not exist in the train_dir folder
    saver = tf.train.Saver()
    save_path = os.path.join(FLAGS.train_dir, FLAGS.filename)
    if os.path.isfile(save_path):
        saver.restore(sess, os.path.join(FLAGS.train_dir, FLAGS.filename))
    else:
        train_params = {
            'nb_epochs': FLAGS.nb_epochs,
            'batch_size': FLAGS.batch_size,
            'learning_rate': FLAGS.learning_rate
        }
        model_train(sess, x, y, predictions, X_train, Y_train,
                    args=train_params)
        saver.save(sess, save_path)

    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': FLAGS.batch_size}
    accuracy = model_eval(sess, x, y, predictions, X_test, Y_test,
                          args=eval_params)
    assert X_test.shape[0] == 10000, X_test.shape
    print('Test accuracy on legitimate test examples: {0}'.format(accuracy))

    ###########################################################################
    # Craft adversarial examples using the Jacobian-based saliency map approach
    ###########################################################################
    print('Crafting ' + str(FLAGS.source_samples) + ' * ' +
          str(FLAGS.nb_classes-1) + ' adversarial examples')

    # This array indicates whether an adversarial example was found for each
    # test set sample and target class
    results = np.zeros((FLAGS.nb_classes, FLAGS.source_samples), dtype='i')

    # This array contains the fraction of perturbed features for each test set
    # sample and target class
    perturbations = np.zeros((FLAGS.nb_classes, FLAGS.source_samples),
                             dtype='f')

    # Define the TF graph for the model's Jacobian
    grads = jacobian_graph(predictions, x, FLAGS.nb_classes)

    # Initialize our array for grid visualization
    grid_shape = (FLAGS.nb_classes,
                  FLAGS.nb_classes,
                  FLAGS.img_rows,
                  FLAGS.img_cols,
                  FLAGS.nb_channels)
    grid_viz_data = np.zeros(grid_shape, dtype='f')
    
    eud = {}
    l1d = {}
    succ = {}

    # Loop over the samples we want to perturb into adversarial examples
    for sample_ind in xrange(0, FLAGS.source_samples):
        # We want to find an adversarial example for each possible target class
        # (i.e. all classes that differ from the label given in the dataset)
        current_class = int(np.argmax(Y_test[FLAGS.starting_index + sample_ind]))
        target_classes = other_classes(FLAGS.nb_classes, current_class)
        
        filename = "pic/%s_jsma.jpg"%(FLAGS.starting_index + sample_ind)
        testImage = np.squeeze(X_test[(FLAGS.starting_index + sample_ind):(FLAGS.starting_index + sample_ind+1)][0])
        save(0,testImage,filename)

        # For the grid visualization, keep original images along the diagonal
        #grid_viz_data[current_class, current_class, :, :, :] = np.reshape(
        #        X_test[sample_ind:(sample_ind+1)],
        #        (FLAGS.img_rows, FLAGS.img_cols, FLAGS.nb_channels))
                
        # initialise data collection
        eud[sample_ind] = 1000.0
        l1d[sample_ind] = 1000.0
        succ[sample_ind] = 0

        # Loop over all target classes
        for target in target_classes:
            print('--------------------------------------')
            print('Creating adv. example for target class ' + str(target))

            # This call runs the Jacobian-based saliency map approach
            adv_x, res, percent_perturb = jsma(sess, x, predictions, grads,
                                               X_test[(FLAGS.starting_index+sample_ind):
                                                      (FLAGS.starting_index+sample_ind+1)],
                                               target, theta=FLAGS.thetaValue, gamma=0.1,
                                               increase=True, back='tf',
                                               clip_min=0, clip_max=1)
                                               
            #print(np.max(adv_x))

            # Display the original and adversarial images side-by-side
            #if FLAGS.viz_enabled:
            #    if 'figure' not in vars():
            #            figure = pair_visual(
            #                    np.reshape(X_test[(FLAGS.starting_index+sample_ind):(FLAGS.starting_index+sample_ind+1)],
            #                               (FLAGS.img_rows, FLAGS.img_cols)),
            #                    np.reshape(adv_x,
            #                               (FLAGS.img_rows, FLAGS.img_cols)))
            #    else:
            #        figure = pair_visual(
            #                np.reshape(X_test[(FLAGS.starting_index+sample_ind):(FLAGS.starting_index+sample_ind+1)],
            #                           (FLAGS.img_rows, FLAGS.img_cols)),
            #                np.reshape(adv_x, (FLAGS.img_rows,
            #                           FLAGS.img_cols)), figure)

            # Add our adversarial example to our grid data
            #grid_viz_data[target, current_class, :, :, :] = np.reshape(
            #        adv_x, (FLAGS.img_rows, FLAGS.img_cols, FLAGS.nb_channels))
                    
            filename = "pic/%s_jsma_%s_%s.jpg"%(FLAGS.starting_index+sample_ind,FLAGS.thetaValue,target)                        
            testImage1 = np.squeeze(adv_x[0])
            fileHandler.write("theta value: %s\n"%(FLAGS.thetaValue))
            fileHandler.write("target: %s\n"%(target))
            fileHandler.write("euclidean distance: %s\n"%(euclideanDistance(testImage1,testImage))) 
            fileHandler.write("L1 distance: %s\n"%(l1Distance(testImage1,testImage)))
            save(0,testImage1,filename)


            # Update the arrays for later analysis
            results[target, sample_ind] = res
            perturbations[target, sample_ind] = percent_perturb
            
            # collect data 
            temp_x = X_test[FLAGS.starting_index+sample_ind]
            adv_x = adv_x[0]
            temp_eud = euclideanDistance(temp_x,adv_x)
            if eud[sample_ind] > temp_eud: 
                eud[sample_ind] = temp_eud
            temp_l1d = l1Distance(temp_x,adv_x)
            if l1d[sample_ind] > temp_l1d: 
                l1d[sample_ind] = temp_l1d  
            if succ[sample_ind] == 0: 
                succ[sample_ind] = res    
                
            #print("res=%s"%(res)) 

    # Compute the number of adversarial examples that were successfuly found
    nb_targets_tried = ((FLAGS.nb_classes - 1) * FLAGS.source_samples)
    succ_rate = float(np.sum(results)) / nb_targets_tried
    print('Avg. rate of successful adv. examples {0:.2f}'.format(succ_rate))

    # Compute the average distortion introduced by the algorithm
    percent_perturbed = np.mean(perturbations)
    print('Avg. rate of perturbed features {0:.2f}'.format(percent_perturbed))

    # Compute the average distortion introduced for successful samples only
    percent_perturb_succ = np.mean(perturbations * (results == 1))
    print('Avg. rate of perturbed features for successful '
          'adversarial examples {0:.2f}'.format(percent_perturb_succ))
          
    # print data 
    for e in eud.keys():
        eud[e] = eud[e] * succ[e] 
    for e in l1d.keys():
        l1d[e] = l1d[e] * succ[e] 
    print("Average Euclidean distance is %s"%(sum(eud.values()) / float(len(eud))))
    print("Average L1 distance is %s"%(sum(l1d.values()) / float(len(l1d))))
    print("Success rate is %s"%(sum(succ.values()) / float(len(succ))))
    

    fileHandler.write("Average Euclidean distance is %s\n"%(sum(eud.values()) / float(len(eud))))
    fileHandler.write("Average L1 distance is %s\n"%(sum(l1d.values()) / float(len(l1d))))
    fileHandler.write("Success rate is %s\n"%(sum(succ.values()) / float(len(succ))))
    fileHandler.close()
    
    # Close TF session
    sess.close()

    # Finally, block & display a grid of all the adversarial examples
    #if FLAGS.viz_enabled:
    #    _ = grid_visual(grid_viz_data)
        
def euclideanDistance(image1,image2):
    distance = 0
    ne = 0 
    if len(image1.shape) == 2:
        for x in range(len(image1)):
            for y in range(len(image1[0])):
                ne += 1
                if image1[x][y] != image2[x][y]: 
                    distance += (image1[x][y] - image2[x][y]) ** 2
    elif len(image1.shape) == 3:
        for x in range(len(image1)):
            for y in range(len(image1[0])):
               for z in range(len(image1[0][0])):
                  ne += 1
                  if image1[x][y][z] != image2[x][y][z]: 
                     distance += (image1[x][y][z] - image2[x][y][z]) ** 2

    elif len(image1.shape) == 1:
        for x in range(len(image1)):
            ne += 1
            if image1[x] != image2[x]: 
                distance += (image1[x] - image2[x]) ** 2

    return math.sqrt(float(distance) / ne)
    
def l1Distance(image1,image2):
    distance = 0
    ne = 0 
    if len(image1.shape) == 2:
        for x in range(len(image1)):
            for y in range(len(image1[0])):
                ne += 1
                if image1[x][y] != image2[x][y]: 
                    distance += math.fabs(image1[x][y] - image2[x][y])
    elif len(image1.shape) == 3:
        for x in range(len(image1)):
            for y in range(len(image1[0])):
               for z in range(len(image1[0][0])):
                  ne += 1
                  if image1[x][y][z] != image2[x][y][z]: 
                     distance += math.fabs(image1[x][y][z] - image2[x][y][z])

    elif len(image1.shape) == 1:
        for x in range(len(image1)):
            ne += 1
            if image1[x] != image2[x]: 
                distance += math.fabs(image1[x] - image2[x])
    #print "distance = %s"%(distance)
    return (float(distance) / ne)


def read(digits=np.arange(10), dataset = "training", path = "."):
    """
    Python function for importing the MNIST data set.
    """
    if dataset is "training":
        fname_img = os.path.join(path, 'mnist/train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'mnist/train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 'mnist/t10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'mnist/t10k-labels-idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in xrange(size) if lbl[k] in digits ]
    N = len(ind)

    images = np.zeros((N, rows, cols), dtype=np.uint8)
    labels = np.zeros((N, 1), dtype=np.int8)
    for i in range(len(ind)):
        images[i] = np.array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels


def save(layer,image,filename):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image * 255, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.savefig(filename)




if __name__ == '__main__':
    app.run()
