'''
Test the Neural Network
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import keras
import pandas
import numpy as np
import localConfig as cfg

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
    parser.add_argument('-o', '--overtrainingCheck', action='store_true', help='Wether there was overtraining')
    parser.add_argument('-p', '--prediction', action='store_true', help='Predictions plot')
    parser.add_argument('-e', '--efficiencyAndFOM', action='store_true', help='Plot efficiency and FOM')
    parser.add_argument('-r', '--areaUnderROC', action='store_true', help='Area under ROC plot')
    parser.add_argument('-w', '--weights', action='store_true', help='Plot neural network weights')
    parser.add_argument('-z', '--structure', action='store_true', help='Plot neural network structure')
    parser.add_argument('-d', '--preview', action='store_true', help='Preview plots')

#python plotNN.py -v -f Model_Ver_3 -b -c -o -p -r -s

    from prepareData import *
    args = parser.parse_args()

    import matplotlib.pyplot as pltl
    from keras.models import model_from_json
    from commonFunctions import assure_path_exists
    from sklearn import metrics

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

    #dataDev["NN"] = model.predict(XDev)
    #dataVal["NN"] = model.predict(XVal)
    #dataTest["NN"] = model.predict(XTest)

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

    print("Accuracy score DEV: {}".format(score[0]))
    print("Accuracy score VAL: {}".format(score[1]))
    print("Accuracy score TEST: {}".format(score[2]))

    f = open(plots_path+"Score.txt","w")
    f.write("Accuracy_score {} {} {}".format(score[0], score[1], score[2]))

    # --- Calculate Classification Log Loss --- #
    score.append(metrics.log_loss(YDev, devPredict,sample_weight=weightDev))
    score.append(metrics.log_loss(YVal, valPredict,sample_weight=weightVal))
    score.append(metrics.log_loss(YTest, testPredict,sample_weight=weightTest))

    print("Log loss score DEV: {}".format(score[3]))
    print("Log loss score DEV: {}".format(score[3]))
    print("Log loss score DEV: {}".format(score[3]))
    f.write("Log_loss_score {} {} {}".format(score[3],score[4],score[5]))
    f.close()

    if args.verbose:
        print("Getting scores ...")

    scoreDev = model.evaluate(XDev, YDev, sample_weight=weightDev, verbose = 0)
    scoreVal = model.evaluate(XVal, YVal, sample_weight=weightVal, verbose = 0)
    scoreTest = model.evaluate(XTest, YTest, sample_weight=weightTest, verbose = 0)
    quit()

    if args.verbose:
        print "Calculating parameters ..."

    sig_dataDev = dataDev[dataDev.category==1];     bkg_dataDev = dataDev[dataDev.category == 0]      # separar sig e bkg em dataDev
    sig_dataVal = dataVal[dataVal.category == 1];    bkg_dataVal = dataVal[dataVal.category == 0]       # separar sig e bkg em dataVal
    sig_dataTest = dataTest[dataTest.category==1];    bkg_dataTest = dataTest[dataTest.category==0]    # separar sig e bkg em dataTest

    if args.allPlots:
        args.loss = True
        args.accuracy = True
        args.overtrainingCheck = True
        args.prediction = True
        args.efficiencyAndFOM = True
        args.areaUnderROC = True
        args.weights = True

    if args.loss:
        import pickle
        loss = pickle.load(open(loss_path+"loss_"+model_name+".pickle", "rb"))
        val_loss = pickle.load(open(loss_path+"val_loss_"+model_name+".pickle", "rb"))
        if args.verbose:
            print "val_loss = ", str(val_loss[-1]), "loss = ", str(loss[-1]), "val_loss - loss = ", str(val_loss[-1]-loss[-1])

        plt.plot(loss)
        plt.plot(val_loss)
        plt.grid()
        plt.title('Model loss')
        plt.ylabel('Loss')
        #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.xlabel('Epoch')
        plt.legend(['train'], loc='best')
        plt.legend(['train', 'val'], loc='best')
        plt.savefig(plots_path+'loss_'+model_name+'.pdf')
        if args.preview:
            plt.show()
        plt.close()

    if args.accuracy:
        import pickle
        acc = pickle.load(open(acc_path+"acc_"+model_name+".pickle", "rb"))
        val_acc = pickle.load(open(acc_path+"val_acc_"+model_name+".pickle", "rb"))
        if args.verbose:
            print "val_acc = ", str(val_acc[-1]), "acc = ", str(acc[-1]), "val_acc - acc = ", str(val_acc[-1]-acc[-1])
        plt.plot(acc)
        plt.plot(val_acc)
        plt.grid()
        plt.ylim(0.8,0.9)
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.xlabel('Epoch')
        plt.legend(['train'], loc='best')
        plt.legend(['train', 'val'], loc='best')
        plt.savefig(plots_path+'acc_'+model_name+'.pdf')
        if args.preview:
            plt.show()
        plt.close()

    # --- Over Training Check --- #

    # Negative bins appear on hist y-axis???
    if args.overtrainingCheck:
        from scipy.stats import ks_2samp
        from sklearn.metrics import cohen_kappa_score

        # Returns: kappa : float
        # The kappa statistic, which is a number between -1 and 1.
        # Scores above .8 are generally considered good agreement; zero or lower means no agreement (practically random labels).
        cohen_kappa=cohen_kappa_score(YTest, dataTest["NN"].round())

        # Computes the Kolmogorov-Smirnov statistic on 2 samples.
        # This is a two-sided test for the null hypothesis that 2 independent samples are drawn from the same continuous distribution.
        # Returns:	D (float) KS statistic
        #           p-value (float) two-tailed p-value
        # Sao comparadas duas amostras de predicao que provem da NN
        km_value=ks_2samp((sig_dataDev["NN"].append(bkg_dataDev["NN"])),(sig_dataTest["NN"].append(bkg_dataTest["NN"]))) # append() does not change sig_dataDev
        if args.verbose:
            print "Cohen Kappa score:", cohen_kappa
            print "KS test statistic:", km_value[0]
            print "KS test p-value:", km_value[1]

        #plt.yscale('log')
        plt.hist(sig_dataDev["NN"], 50, facecolor='blue', alpha=0.7, normed=1, weights=sig_dataDev["EventWeight"]) # histtype by default is "bar"
        plt.hist(bkg_dataDev["NN"], 50, facecolor='red', alpha=0.7, normed=1, weights=bkg_dataDev["EventWeight"])
        plt.hist(sig_dataTest["NN"], 50, color='blue', alpha=1, normed=1, histtype="step", weights=sig_dataTest["EventWeight"]) # "step" generates a lineplot that is by default unfilled.
        plt.hist(bkg_dataTest["NN"], 50, color='red', alpha=1, normed=1, histtype="step",weights=bkg_dataTest["EventWeight"])
        plt.grid()
        plt.xlabel('NN output')
        plt.suptitle("MVA overtraining check for classifier: NN", fontsize=13, fontweight='bold') # MVA = MultiVariable Analysis
        plt.title("Cohen's kappa: {0}\nKolmogorov Smirnov test (p_value): {1}".format(cohen_kappa, km_value[1]), fontsize=10)
        plt.legend(['Signal (Train sample)', 'Background (Train sample)', 'Signal (Test sample)', 'Background (Test sample)'], loc='best')
        plt.savefig(plots_path+'hist_'+model_name+'.pdf', bbox_inches='tight')
        if args.preview:
            plt.show()
        plt.close()

    # --- Predictions plot --- #
    # Negative bins appear on hist y-axis???
    if args.prediction:
        both_dataDev = bkg_dataDev.append(sig_dataDev)
        plt.figure(figsize=(7,6))
        plt.xlabel('NN output')
        plt.title("Number of Events")
        #plt.yscale('log', nonposy='clip')
        plt.hist(bkg_dataDev["NN"], 50, facecolor='red', normed=1, weights=bkg_dataDev["EventWeight"]) # in original code there is not normalization but the plot seems to be normalized ???
        plt.hist(both_dataDev["NN"], 50, color="blue", histtype="step", normed=1, weights=both_dataDev["EventWeight"])
        plt.legend(['Background + Signal (test sample)', 'Background (test sample)'], loc="best" )
        plt.grid()
        plt.savefig(plots_path+'pred_'+model_name+'.pdf', bbox_inches='tight')
        if args.preview:
            plt.show()
        plt.close()

    # PLOTTING FOM AND Efficiency
    if args.efficiencyAndFOM:
        from commonFunctions import FullFOM, getYields

        fomEvo = []
        fomCut = []

        bkgEff = []
        sigEff = []

        sig_Init = sig_dataTest.EventWeight.sum() * luminosity * 3
        bkg_Init =  bkg_dataTest.EventWeight.sum() * luminosity * 3

        for cut in np.arange(0.0, 0.9999, 0.001):
            # return ((sigYield, sigYieldUnc), (bkgYield, bkgYieldUnc))
            sig, bkg = getYields(dataTest, cut=cut, luminosity=luminosity)
            if sig[0] > 0 and bkg[0] > 0:
                fom, fomUnc = FullFOM(sig, bkg) # return (fom, fomErr)
                fomEvo.append(fom)
                fomCut.append(cut)
                bkgEff.append(bkg[0]/bkg_Init) # bkg efficiency ???
                sigEff.append(sig[0]/sig_Init) # sig efficiency ???

        max_FOM=0.0

        for k in fomEvo:
            if k>max_FOM:
                max_FOM=k

        # SAVE VALUES OF FOM EVO AND CUT TO DO A FOM SUMMARY
        f= open(plots_path+"FOM_evo_data.txt","w+")
        f.write("\n".join(map(str,fomEvo)))
        f.close()

        f= open(plots_path+"FOM_cut_data.txt","w+")
        f.write("\n".join(map(str,fomCut)))
        f.close()

        Eff = zip(bkgEff, sigEff)

        if args.verbose:
            print "Maximized FOM:", max_FOM
            if max_FOM != 0.0:
                print "FOM Cut:", fomCut[fomEvo.index(max_FOM)]
            else:
                print "ERROR: An unexpected Value: max_FOM == 0.0"

            # return ((sigYield, sigYieldUnc), (bkgYield, bkgYieldUnc))
            tmpSig, tmpBkg = getYields(dataTest)
            sigYield, sigYieldUnc = tmpSig
            bkgYield, bkgYieldUnc = tmpBkg

            selectedTest = dataTest[dataTest.NN>fomCut[fomEvo.index(max_FOM)]]
            selectedSig = selectedTest[selectedTest.category == 1]
            selectedBkg = selectedTest[selectedTest.category == 0]
            sigYield = selectedSig.EventWeight.sum() * luminosity * 3  #The factor 3 comes from the splitting
            bkgYield = selectedBkg.EventWeight.sum() * luminosity * 3

            print "Selected events left after cut @", fomCut[fomEvo.index(max_FOM)]
            print "   Number of selected Signal Events:", len(selectedSig)
            print "   Number of selected Background Events:", len(selectedBkg)
            print "   Sig Yield:", sigYield
            print "   Bkg Yield:", bkgYield

        plt.figure(figsize=(7,6))
        plt.subplots_adjust(hspace=0.5)

        plt.subplot(211)
        plt.plot(fomCut, fomEvo, linewidth = 0.5)
        plt.title("FOM")
        plt.ylabel("FOM")
        plt.xlabel("ND")    # O que significa ND ???
        plt.legend(["Max. FOM: {0}".format(max_FOM)], loc='best')
        plt.grid()

        plt.subplot(212)
        plt.semilogy(fomCut, Eff , linewidth = 0.5) # Eff = zip(bkgEff, sigEff)

        # axvspan(xmin, xmax, ymin=0, ymax=1, **kwargs)
        # Draw a vertical span (rectangle) from xmin to xmax. With the default values of ymin = 0 and ymax = 1.
        plt.axvspan(fomCut[fomEvo.index(max_FOM)], 1, facecolor='#2ca02c', alpha=0.3)

        # axvline(x=0, ymin=0, ymax=1, **kwargs)
        # Add a vertical line across the axes.
        plt.axvline(x=fomCut[fomEvo.index(max_FOM)], ymin=0, ymax=1)
        plt.title("Efficiency")
        plt.ylabel("Eff")
        plt.xlabel("ND")
        plt.legend(['Background', 'Signal'], loc='best')
        plt.grid()

        plt.savefig(plots_path+'FOM_EFF_'+model_name+'.pdf', bbox_inches='tight')
        if args.preview:
            plt.show()
        plt.close()

        #SAME BUT ZOOMED IN , NO LOG yscale
        plt.figure(figsize=(7,6))
        plt.subplots_adjust(hspace=0.5)

        plt.subplot(211)
        plt.plot(fomCut, fomEvo, linewidth = 0.3)
        plt.xlim(0.88, 1.01)
        plt.xticks(np.arange(0.88 , 1.01, step = 0.01))
        plt.title("FOM")
        plt.ylabel("FOM")
        plt.xlabel("ND")
        plt.legend(["Max. FOM: {0}".format(max_FOM)], loc='best')
        plt.grid()

        plt.subplot(212)
        plt.plot(fomCut, Eff , linewidth = 0.3)
        plt.xlim(0.88 , 1.01)

        # Get or set the current tick locations and labels of the x-axis.
        plt.xticks(np.arange(0.88 , 1.01, step = 0.01))
        plt.axvspan(fomCut[fomEvo.index(max_FOM)], 1, facecolor='#2ca02c', alpha=0.3)
        plt.axvline(x=fomCut[fomEvo.index(max_FOM)], ymin=0, ymax=1)
        plt.title("Efficiency")
        plt.ylabel("Eff")
        plt.xlabel("ND")
        plt.legend(['Background', 'Signal'], loc='best')
        plt.grid()

        plt.savefig(plots_path+'FOM_EFF_zoomed_'+model_name+'.pdf', bbox_inches='tight')
        if args.preview:
            plt.show()
        plt.close()

    # PLOTTING the ROC function
    if args.areaUnderROC:
        from sklearn.metrics import roc_auc_score, roc_curve

        # roc_auc_score(y_true, y_score, average='macro', sample_weight=None, max_fpr=None)
        # Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
        # Returns: auc (float)
        roc_integralDev = roc_auc_score(dataDev.category, dataDev.NN)
        roc_integralVal = roc_auc_score(dataVal.category, dataVal.NN)
        roc_integralTest = roc_auc_score(dataTest.category, dataTest.NN) # sample_weight = dataTest.EventWeight ???

        # roc_curve(y_true, y_score, pos_label=None, sample_weight=None, drop_intermediate=True)
        # Compute Receiver operating characteristic (ROC)
        # Returns:
        #           fpr : array, shape = [>2]
        #           tpr : array, shape = [>2]
        #           thresholds : array, shape = [n_thresholds]
        fprDev, tprDev, _Dev = roc_curve(dataDev.category, dataDev.NN)
        fprVal, tprVal, _Val = roc_curve(dataVal.category, dataVal.NN)
        fprTest, tprTest, _Test = roc_curve(dataTest.category, dataTest.NN)
        if args.verbose:
            print "ROC Curve IntegralDev:", roc_integralDev
            print "ROC Curve IntegralVal:", roc_integralVal
            print "ROC Curve IntegralTest:", roc_integralTest

        plt.figure()
        plt.subplots_adjust(hspace=0.5)

        plt.subplot(211)
        plt.plot(fprDev, tprDev, '--')
        plt.plot(fprVal, tprVal, ':')
        plt.plot(fprTest, tprTest, linewidth=0.5)
        plt.grid()
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        rocLegend = ["Dev Integral: {0}".format(roc_integralDev),"Val Integral: {0}".format(roc_integralVal),"Test Integral: {0}".format(roc_integralTest)]
        plt.legend(rocLegend, loc='best')
        # plt.savefig(plots_path+'ROC_'+model_name+'.pdf', bbox_inches='tight')
        # if args.preview:
        #     plt.show()
        # plt.close()

        #PLOTTING ROCK ZOOMED
        plt.subplot(212)
        plt.plot(fprDev, tprDev, '--')
        plt.plot(fprVal, tprVal, ':')
        plt.plot(fprTest, tprTest, linewidth=0.5)
        plt.grid()
        plt.xlim(0 , 0.3)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve ZOOMED')
        rocLegend = ["Dev Integral: {0}".format(roc_integralDev),"Val Integral: {0}".format(roc_integralVal),"Test Integral: {0}".format(roc_integralTest)]
        plt.legend(rocLegend, loc='best')
        plt.savefig(plots_path+'ROC_ROC_zoomed_'+model_name+'.pdf', bbox_inches='tight')
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

        figure = plt.figure()
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
        plt.savefig(plots_path+'Weights_'+model_name+'.pdf', bbox_inches='tight')
        if args.preview:
            plt.show()
