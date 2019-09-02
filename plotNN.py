'''
Test the Neural Network
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import keras
import pandas
import numpy as np
import localConfig as cfg
import matplotlib.pyplot as plt

# Plot a confusion matrix. cm is the confusion matrix, names are the names of the classes.
def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

if __name__ == "__main__":
    import sys
    import argparse

    ## Input arguments. Pay speciall attention to the required ones.
    parser = argparse.ArgumentParser(description='Process the command line options')
    parser.add_argument('-v', '--verbose', action='store_true', help='Whether to print verbose output')
    parser.add_argument('-f', '--file',type=str, required=True, help='File name')
    parser.add_argument('-a', '--allPlots', action='store_true', help='Wether to plot all graphs')
    parser.add_argument('-b', '--loss', action='store_true', help='Loss plot')
    parser.add_argument('-c', '--accuracy', action='store_true', help='Accuracy plot')
    parser.add_argument('-r', '--areaUnderROC', action='store_true', help='Area under ROC plot')
    parser.add_argument('-w', '--weights', action='store_true', help='Plot neural network weights')

    parser.add_argument('-cm', '--confusionMatrix', action='store_true', help='Plot confusion matrix')
    parser.add_argument('-d', '--preview', action='store_true', help='Preview plots')

#python plotNN.py -v -f Model_Ver_3 -b -c -o -p -r -s

    from prepareData import *
    args = parser.parse_args()

    from keras.models import model_from_json
    from commonFunctions import assure_path_exists
    from sklearn import metrics
    from sklearn.metrics import confusion_matrix
    from matplotlib.backends.backend_pdf import PdfPages

    classes = ["sig","stop","ttbar","WlvZqq","WqqWlv","W+Jets"]
    if args.file != None:
        model_name = args.file
        # lgbk = "/home/t3atlas/ev19u056/mc_classification/"
        filepath = cfg.lgbk + "test/" + model_name
        loss_path = filepath + "/loss/"
        acc_path = filepath + "/accuracy/"
    else:
        print "ERROR: Missing filename"
        quit()

    f=open(filepath + "/prepareData_" + model_name + ".txt", "r")
    fraction = float(f.readline())

    dataDev, dataVal, dataTest, XDev, YDev, weightDev, XVal, YVal, weightVal, XTest, YTest, weightTest = dataLoader(filepath+"/", model_name, fraction)
    os.chdir(filepath+"/")
    plots_path = filepath+"/plots_"+model_name+"/"
    assure_path_exists(plots_path)

    if args.verbose:
        print "Loading Model ..."

    ## Load your trainned model
    with open(model_name+'.json', 'r') as json_file:
      loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_name+".h5")
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

    if args.verbose:
        print("Getting predictions ...")

    # numpy.argmax(a, axis=None, out=None) => Returns the indices of the maximum values along an axis
    devPredict = model.predict(XDev)
    valPredict = model.predict(XVal)
    testPredict = model.predict(XTest)

    dataDev["NN"] = np.argmax(devPredict,axis=1) # raw probabilities to chosen class (highest probability)
    dataVal["NN"] = np.argmax(valPredict,axis=1)
    dataTest["NN"] = np.argmax(testPredict,axis=1)

    score = []
    score.append(metrics.accuracy_score(np.argmax(YDev,axis=1), dataDev["NN"],sample_weight=weightDev))
    score.append(metrics.accuracy_score(np.argmax(YVal,axis=1), dataVal["NN"],sample_weight=weightVal))
    score.append(metrics.accuracy_score(np.argmax(YTest,axis=1), dataTest["NN"],sample_weight=weightTest))

    if args.verbose:
        print("Accuracy score DEV: {}".format(score[0]))
        print("Accuracy score VAL: {}".format(score[1]))
        print("Accuracy score TEST: {}".format(score[2]))

    f = open(plots_path+"Score.txt","w")
    f.write("Accuracy_score {} {} {}\n".format(score[0], score[1], score[2]))

    # --- Calculate Classification Log Loss --- #
    score.append(metrics.log_loss(YDev, devPredict)#,sample_weight=weightDev))
    score.append(metrics.log_loss(YVal, valPredict)#,sample_weight=weightVal))
    score.append(metrics.log_loss(YTest, testPredict)#,sample_weight=weightTest))

    if args.verbose:
        print("Log loss score DEV: {}".format(score[3]))
        print("Log loss score VAL: {}".format(score[4]))
        print("Log loss score TEST: {}".format(score[5]))
    f.write("Log_loss_score {} {} {}\n".format(score[3],score[4],score[5]))
    f.close()

    # if args.verbose:
    #     print "Calculating parameters ..."
    #
    # sig_dataDev = dataDev[dataDev.category==1];     bkg_dataDev = dataDev[dataDev.category == 0]      # separar sig e bkg em dataDev
    # sig_dataVal = dataVal[dataVal.category == 1];    bkg_dataVal = dataVal[dataVal.category == 0]       # separar sig e bkg em dataVal
    # sig_dataTest = dataTest[dataTest.category==1];    bkg_dataTest = dataTest[dataTest.category==0]    # separar sig e bkg em dataTest

    if args.allPlots:
        args.loss = True
        args.accuracy = True
        args.areaUnderROC = True
        args.weights = True
        args.confusionMatrix = True

    # PLOTTING the ROC function
    if args.areaUnderROC:
        from sklearn.metrics import roc_auc_score, roc_curve, auc
        from sklearn.preprocessing import label_binarize
        from scipy import interp
        from itertools import cycle

        # Compute ROC curve and ROC area for each class
        fprTest = dict()
        tprTest = dict()
        roc_auc_Test = dict()
        n_classes = 6
        for i in range(n_classes):
            fprTest[i], tprTest[i], _ = roc_curve(YTest[:, i], testPredict[:, i])
            roc_auc_Test[i] = auc(fprTest[i], tprTest[i]) # Compute Area Under the Curve (AUC) using the trapezoidal rule

        # Compute micro-average ROC curve and ROC area
        fprTest["micro"], tprTest["micro"], _ = roc_curve(YTest.ravel(), testPredict.ravel())
        roc_auc_Test["micro"] = auc(fprTest["micro"], tprTest["micro"])

        ##############################################################################
        # Plot ROC curves for the multiclass problem
        # First aggregate all false positive rates
        all_fprTest = np.unique(np.concatenate([fprTest[i] for i in range(n_classes)])) # Returns the sorted unique elements of an array

        # Then interpolate all ROC curves at this points
        mean_tprTest = np.zeros_like(all_fprTest) # Return an array of zeros with the same shape and type as a given array.
        for i in range(n_classes):
            # numpy.interp(x, xp, fp, left=None, right=None, period=None)
            # Returns the one-dimensional piecewise linear interpolant to a function with given discrete data points (xp, fp), evaluated at x.
            mean_tprTest += interp(all_fprTest, fprTest[i], tprTest[i])

        # Finally average it and compute AUC
        mean_tprTest /= n_classes

        fprTest["macro"] = all_fprTest
        tprTest["macro"] = mean_tprTest
        roc_auc_Test["macro"] = auc(fprTest["macro"], tprTest["macro"])

        # Plot all ROC curves
        pdf_pages = PdfPages(plots_path+"ROC_"+model_name+".pdf") # plots_path = filepath+"/plots_"+model_name+"/"
        fig = plt.figure(figsize=(8.27, 5.845), dpi=100)
        lw = 1 # linewidth
        plt.plot(fprTest["micro"], tprTest["micro"], label='micro-average (area = {0:0.4f})'.format(roc_auc_Test["micro"]), color='deeppink', linestyle=':', linewidth=4)
        plt.grid()
        plt.plot(fprTest["macro"], tprTest["macro"], label='macro-average (area = {0:0.4f})'.format(roc_auc_Test["macro"]), color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue','black','brown','darkgreen'])
        for i, color in zip(range(n_classes), colors):
            #plt.plot(fprTest[i], tprTest[i], color=color, lw=lw, label='class {0} (area = {1:0.4f})'.format(i, roc_auc_Test[i]))
            plt.plot(fprTest[i], tprTest[i], color=color, lw=lw, label='{0} (area = {1:0.4f})'.format(classes[i], roc_auc_Test[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve for multi-class')
        plt.legend(loc="lower right")
        pdf_pages.savefig(fig)
        if args.preview:
            plt.show()
        plt.close()

        # # roc_auc_score(y_true, y_score, average='macro', sample_weight=None, max_fpr=None)
        # # Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
        # # Returns: auc (float)
        # roc_integralDev = roc_auc_score(dataDev.category, dataDev.NN)
        # roc_integralVal = roc_auc_score(dataVal.category, dataVal.NN)
        # roc_integralTest = roc_auc_score(dataTest.category, dataTest.NN) # sample_weight = dataTest.EventWeight ???
        #
        # # roc_curve(y_true, y_score, pos_label=None, sample_weight=None, drop_intermediate=True)
        # # Compute Receiver operating characteristic (ROC)
        # # Returns:
        # #           fpr : array, shape = [>2]
        # #           tpr : array, shape = [>2]
        # #           thresholds : array, shape = [n_thresholds]
        # # Note: this implementation is restricted to the binary classification task.
        # fprDev, tprDev, _Dev = roc_curve(dataDev.category, dataDev.NN)
        # fprVal, tprVal, _Val = roc_curve(dataVal.category, dataVal.NN)
        # fprTest, tprTest, _Test = roc_curve(dataTest.category, dataTest.NN)

    if args.confusionMatrix:
        # Compute confusion matrix
        cm = confusion_matrix(np.argmax(YTest,axis=1),dataTest["NN"])
        if args.verbose:
            print('Confusion matrix, without normalization')
            print(cm)

        pdf_pages = PdfPages(plots_path+"ConfusionMatrix_"+model_name+".pdf") # plots_path = filepath+"/plots_"+model_name+"/"
        fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
        plt.subplots_adjust(hspace=0.5)
        plt.subplot(2,1,1)
        #samples = ['0','1','2','3','4','5']
        plot_confusion_matrix(cm, classes)

        # Normalize the confusion matrix by row (i.e by the number of samples in each class)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        if args.verbose:
            print('Normalized confusion matrix...')
            np.set_printoptions(precision=2)
            print(cm_normalized)

        plt.subplot(2,1,2)
        plot_confusion_matrix(cm_normalized, classes, title='Normalized confusion matrix')
        pdf_pages.savefig(fig)
        if args.preview:
            plt.show()
        plt.close()

    if args.loss:
        import pickle
        loss = pickle.load(open(loss_path+"loss_"+model_name+".pickle", "rb"))
        val_loss = pickle.load(open(loss_path+"val_loss_"+model_name+".pickle", "rb"))
        if args.verbose:
            print "val_loss = ", str(val_loss[-1]), "loss = ", str(loss[-1]), "val_loss - loss = ", str(val_loss[-1]-loss[-1])

        pdf_pages = PdfPages(plots_path+'loss_'+model_name+".pdf") # plots_path = filepath+"/plots_"+model_name+"/"
        fig = plt.figure(figsize=(8.27, 5.845), dpi=100)
        plt.plot(loss, label='train = {0:0.4f}'.format(loss[-1]))
        plt.plot(val_loss, label='val = {0:0.4f}'.format(val_loss[-1]))
        plt.grid()
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        pdf_pages.savefig(fig)
        if args.preview:
            plt.show()
        plt.close()

    if args.accuracy:
        import pickle
        acc = pickle.load(open(acc_path+"acc_"+model_name+".pickle", "rb"))
        val_acc = pickle.load(open(acc_path+"val_acc_"+model_name+".pickle", "rb"))
        if args.verbose:
            print "val_acc = ", str(val_acc[-1]), "acc = ", str(acc[-1]), "val_acc - acc = ", str(val_acc[-1]-acc[-1])

        pdf_pages = PdfPages(plots_path+'acc_'+model_name+".pdf") # plots_path = filepath+"/plots_"+model_name+"/"
        fig = plt.figure(figsize=(8.27, 5.845), dpi=100)
        plt.plot(acc, label='train = {0:0.4f}'.format(acc[-1]))
        plt.plot(val_acc, label='val = {0:0.4f}'.format(val_acc[-1]))
        plt.grid()
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        pdf_pages.savefig(fig)
        if args.preview:
            plt.show()
        plt.close()

    if args.weights:
        import math
        from matplotlib.colors import LinearSegmentedColormap
        #Color maps
        cdict = {'red':   ((0.0, 0.97, 0.97),
                           (0.25, 0.0, 0.0),
                           (0.75, 0.0, 0.0),
                           (1.0, 1.0, 1.0)),

                 'green': ((0.0, 0.25, 0.25),
                           (0.25, 0.15, 0.15),
                           (0.75, 0.39, 0.39),
                           (1.0, 0.78, 0.78)),

                 'blue':  ((0.0, 1.0, 1.0),
                           (0.25, 0.65, 0.65),
                           (0.75, 0.02, 0.02),
                           (1.0, 0.0, 0.0))
                }
        myColor = LinearSegmentedColormap('myColorMap', cdict)
        nLayers = 0
        for layer in model.layers:
            if len(layer.get_weights()) == 0:
                continue
            nLayers+=1

        maxWeights = 0
        pdf_pages = PdfPages(plots_path+'Weights_'+model_name+'.pdf') # plots_path = filepath+"/plots_"+model_name+"/"
        figure = plt.figure(figsize=(8.27, 11.69), dpi=100)
        figure.suptitle("Weights", fontsize=12)

        i=1
        nRow=2
        nCol=3
        if nLayers < 5:
            nRow = 2.0
            nCol = 2
        elif nLayers < 10:
            nRow = math.ceil(nLayers / 3)
            nCol = 3
        else:
            nRow = math.ceil(nLayers / 4)
            nCol = 4

        for layer in model.layers:
            if len(layer.get_weights()) == 0:
                continue
            ax = figure.add_subplot(nRow, nCol,i)
            im = plt.imshow(layer.get_weights()[0], interpolation="none", vmin=-2, vmax=2, cmap=myColor)
            plt.title(layer.name, fontsize=10)
            plt.xlabel("Neuron", fontsize=9)
            plt.ylabel("Input", fontsize=9)
            plt.colorbar(im, use_gridspec=True)
            i+=1

        plt.tight_layout()
        pdf_pages.savefig(figure)
        if args.preview:
            plt.show()
        plt.close()
