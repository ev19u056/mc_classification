"""
Prepare Data for processing
"""
#!/usr/bin/env python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn import decomposition
from keras.utils import np_utils
import pandas as pd
import localConfig as cfg

trainFeatures = ['nFats', 'nJets', 'nTags', 'nTaus', 'nMuons', 'nbJets', 'FJ1nTags', 'nFwdJets', 'nSigJets', 'nElectrons', 'mB1', 'mB2', 'mBB', 'mJ3', 'mL1', 'mTW', 'mVH', 'met', 'pTW', 'FJ1M', 'dRBB', 'mBBJ', 'mVFJ', 'pTB1', 'pTB2', 'pTBB', 'pTJ3', 'ptL1', 'FJ1C2', 'FJ1D2', 'FJ1Pt', 'etaB1', 'etaB2', 'etaBB', 'etaJ3', 'etaL1', 'pTBBJ', 'FJ1Ang', 'FJ1Eta', 'FJ1Phi', 'FJ1T21', 'dEtaBB', 'dPhiBB', 'metSig', 'FJ1KtDR', 'dPhiVBB', 'dPhiVFJ', 'MV2c10B1', 'MV2c10B2', 'metSig_PU', 'mindPhilepB', 'metOverSqrtHT', 'metOverSqrtSumET']
luminosity = 139500 # pb^-1

def chunkReader(tmp):
    result = pd.DataFrame()
    for chunk in tmp:
        chunk.dropna(axis=0,how='any',subset=trainFeatures, inplace=True) # Dropping all rows with any NaN value
        result = result.append(chunk)
    return result

def dataLoader(filepath, name, fraction, luminosity=139500):
    # "/home/t3atlas/ev19u056/projetoWH/data"
    datapath = cfg.loc+'/'

    if fraction > 1.0 or fraction <= 0.0:
        raise ValueError("An invalid fraction was chosen")

    # fix random seed for reproducibility
    seed = 7

    #useSF=True
    useSF=False
    otherFeatures = ['PUWeight','flavB1', 'flavB2', 'EventNumber', 'EventRegime', 'AverageMu', 'EventWeight', 'Sample', 'Description', 'EventFlavor', 'TriggerSF', 'ActualMuScaled', 'AverageMuScaled', 'eventFlagMerged/l','eventFlagResolved/l','BTagSF','ActualMu','LeptonSF', 'phiW', 'phiB1', 'phiB2', 'phiBB', 'phiJ3', 'phiL1']
    trainFeatures = ['nFats', 'nJets', 'nTags', 'nTaus', 'nMuons', 'nbJets', 'FJ1nTags', 'nFwdJets', 'nSigJets', 'nElectrons', 'mB1', 'mB2', 'mBB', 'mJ3', 'mL1', 'mTW', 'mVH', 'met', 'pTW', 'FJ1M', 'dRBB', 'mBBJ', 'mVFJ', 'pTB1', 'pTB2', 'pTBB', 'pTJ3', 'ptL1', 'FJ1C2', 'FJ1D2', 'FJ1Pt', 'etaB1', 'etaB2', 'etaBB', 'etaJ3', 'etaL1', 'pTBBJ', 'FJ1Ang', 'FJ1Eta', 'FJ1Phi', 'FJ1T21', 'dEtaBB', 'dPhiBB', 'metSig', 'FJ1KtDR', 'dPhiVBB', 'dPhiVFJ', 'MV2c10B1', 'MV2c10B2', 'metSig_PU', 'mindPhilepB', 'metOverSqrtHT', 'metOverSqrtSumET']
    usecols = trainFeatures[:]; usecols.append("EventWeight")
    scalingFeatures = ['mB1', 'mB2', 'mBB', 'mJ3', 'mL1', 'mTW', 'mVH', 'met', 'pTW', 'FJ1M', 'dRBB', 'mBBJ', 'mVFJ', 'pTB1', 'pTB2', 'pTBB', 'pTJ3', 'ptL1', 'FJ1C2', 'FJ1D2', 'FJ1Pt', 'etaB1', 'etaB2', 'etaBB', 'etaJ3', 'etaL1', 'pTBBJ', 'FJ1Ang', 'FJ1Eta', 'FJ1Phi', 'FJ1T21', 'dEtaBB', 'dPhiBB', 'metSig', 'FJ1KtDR', 'dPhiVBB', 'dPhiVFJ', 'MV2c10B1', 'MV2c10B2', 'metSig_PU', 'mindPhilepB', 'metOverSqrtHT', 'metOverSqrtSumET']

    nrows_signal = 991141;      XS_signal = 1.37 # pb
    nrows_stopWt = 277816;      XS_stopWt = 71.7
    nrows_ttbar = 4168037;      XS_ttbar = 452.36
    nrows_Wjets = 16650877;     XS_Wjets = 1976.49
    nrows_WlvZqq = 188395;      XS_WlvZqq = 11.413
    nrows_WqqWlv = 334495;      XS_WqqWlv = 50.64

    ttbar_fraction = 1.0/10.0
    WJets_fraction = 1.0/40.0
    chunksize = 1000

    start = time.time()
    print("Reading -> 'qqWlvHbbJ_PwPy8MINLO_ade.csv'")
    tmp = pd.read_csv(datapath+'qqWlvHbbJ_PwPy8MINLO_ade.csv',chunksize=chunksize,nrows = int(nrows_signal*fraction),usecols=usecols)
    df_signal = chunkReader(tmp)

    print("Reading -> 'stopWt_PwPy8_ade.csv'")
    df_stopWt = pd.read_csv(datapath+'stopWt_PwPy8_ade.csv',nrows = int(nrows_stopWt*fraction),usecols=usecols)

    print("Reading -> 'ttbar_nonallhad_PwPy8_ade.csv'")
    tmp = pd.read_csv(datapath+'ttbar_nonallhad_PwPy8_ade.csv',chunksize=chunksize,nrows = int((nrows_ttbar*ttbar_fraction)*fraction),usecols=usecols)
    df_ttbar = chunkReader(tmp)

    print("Reading -> 'WlvZqq_Sh221_ade.csv'")
    df_WlvZqq = pd.read_csv(datapath+'WlvZqq_Sh221_ade.csv',nrows = int(nrows_WlvZqq*fraction),usecols=usecols)

    print("Reading -> 'WqqWlv_Sh221_ade.csv'")
    df_WqqWlv = pd.read_csv(datapath+'WqqWlv_Sh221_ade.csv',nrows = int(nrows_WqqWlv*fraction),usecols=usecols)

    print("Reading -> 'WJets_Sh221.csv'")
    df_WJets = pd.read_csv(datapath+'WJets_Sh221.csv',nrows = int((nrows_Wjets*WJets_fraction)*(fraction)),usecols=usecols)

    df_signal["category"] = 0
    df_stopWt["category"] = 1
    df_ttbar["category"] = 2
    df_WlvZqq["category"] = 3
    df_WqqWlv["category"] = 4
    df_WJets["category"] = 5

    df_signal["sampleWeight"] = 1
    df_stopWt["sampleWeight"] = 1
    df_ttbar["sampleWeight"] = 1
    df_WlvZqq["sampleWeight"] = 1
    df_WqqWlv["sampleWeight"] = 1
    df_WJets["sampleWeight"] = 1

    if fraction < 1.0:
        df_signal.EventWeight = df_signal.EventWeight/fraction
        df_stopWt.EventWeight = df_stopWt.EventWeight/fraction
        df_ttbar.EventWeight = df_ttbar.EventWeight/(fraction*ttbar_fraction)
        df_WlvZqq.EventWeight = df_WlvZqq.EventWeight/fraction
        df_WqqWlv.EventWeight = df_WqqWlv.EventWeight/fraction
        df_WJets.EventWeight = df_WJets.EventWeight/(fraction*WJets_fraction)

    if not useSF:
        df_signal.sampleWeight = df_signal.EventWeight
        df_stopWt.sampleWeight = df_stopWt.EventWeight
        df_ttbar.sampleWeight = df_ttbar.EventWeight
        df_WlvZqq.sampleWeight = df_WlvZqq.EventWeight
        df_WqqWlv.sampleWeight = df_WqqWlv.EventWeight
        df_WJets.sampleWeight = df_WJets.EventWeight

    else:
        scale = fraction if fraction < 1.0 else 1.0
        df_signal.sampleWeight = 1/(nrows_signal*scale)
        df_stopWt.sampleWeight = df_stopWt.sampleWeight*XS_stopWt/(nrows_stopWt*scale)
        df_ttbar.sampleWeight = df_ttbar.sampleWeight*XS_ttbar/(nrows_ttbar*scale*ttbar_fraction)
        df_WlvZqq.sampleWeight = df_WlvZqq.sampleWeight*XS_WlvZqq/(nrows_WlvZqq*scale)
        df_WqqWlv.sampleWeight = df_WqqWlv.sampleWeight*XS_WqqWlv/(nrows_WqqWlv*scale)
        df_WJets.sampleWeight = df_WlvZqq.sampleWeight*XS_Wjets/(nrows_Wjets*scale*WJets_fraction)

    df_signal.sampleWeight = df_signal.sampleWeight/df_signal.sampleWeight.sum()
    df_stopWt.sampleWeight = df_stopWt.sampleWeight/df_stopWt.sampleWeight.sum()
    df_ttbar.sampleWeight = df_ttbar.sampleWeight/df_ttbar.sampleWeight.sum()
    df_WlvZqq.sampleWeight = df_WlvZqq.sampleWeight/df_WlvZqq.sampleWeight.sum()
    df_WqqWlv.sampleWeight = df_WqqWlv.sampleWeight/df_WqqWlv.sampleWeight.sum()
    df_WJets.sampleWeight = df_WJets.sampleWeight/df_WJets.sampleWeight.sum()

    data = None
    for tmp in [df_signal,df_stopWt,df_ttbar,df_WlvZqq,df_WqqWlv,df_WJets]:
            if data is None:
                data = pd.DataFrame(tmp)
                del tmp
            else:
                data = data.append(pd.DataFrame(tmp),ignore_index=True)
                del tmp

    del df_stopWt, df_ttbar, df_WlvZqq, df_WqqWlv, df_WJets, df_signal
    print 'Datasets contain a total of', len(data), ' events'

    # np.utils.to_categorical is used to convert array of labeled data(from 0 to nb_classes-1) to one-hot vector.
    data.category = np_utils.to_categorical(data.category, num_classes=6)

    dataDev, dataVal, dataTest = np.split(data.sample(frac=1,random_state=seed).reset_index(drop=True), [int(0.8*len(data)), int(0.9*len(data))])
    del data

    XDev = dataDev[trainFeatures]
    YDev = dataDev.category
    print(YDev)
    weightDev = np.ravel(dataDev.sampleWeight)

    XVal = dataVal[trainFeatures]
    YVal = dataVal.category
    weightVal = np.ravel(dataVal.sampleWeight)

    XTest = dataTest[trainFeatures]
    YTest = dataTest.category
    weightTest = np.ravel(dataTest.sampleWeight)

    # Creating a text file where all of the prepareData caracteristics are displayed
    f=open(filepath + "prepareData_" + name + ".txt", "w")
    f.write("{}\n".format(fraction))
    f.write("qqWlvHbbJ_PwPy8MINLO_ade.csv:  {} lines     {} \n".format(int(nrows_signal*fraction), fraction))
    f.write("stopWt_PwPy8_ade.csv:          {} lines     {} \n".format(int(nrows_stopWt*fraction), fraction))
    f.write("ttbar_nonallhad_PwPy8_ade.csv: {} lines     {} \n".format(int((nrows_ttbar*ttbar_fraction)*fraction), ttbar_fraction*fraction))
    f.write("WlvZqq_Sh221_ade.csv:          {} lines     {} \n".format(int(nrows_WlvZqq*fraction), fraction))
    f.write("WqqWlv_Sh221_ade.csv:          {} lines     {} \n".format(int(nrows_WqqWlv*fraction), fraction))
    f.write("WJets_Sh221.csv:               {} lines     {} \n".format(int((nrows_Wjets*WJets_fraction)*fraction), WJets_fraction*fraction))

    print "Fitting the scaler and scaling the input variables ..."
    '''
    scaler = StandardScaler().fit(XDev[scalingFeatures])
    XDev[scalingFeatures] = scaler.transform(XDev[scalingFeatures])
    XVal[scalingFeatures] = scaler.transform(XVal[scalingFeatures])
    XTest[scalingFeatures] = scaler.transform(XTest[scalingFeatures])
    '''
    scaler = StandardScaler().fit(XDev)
    XDev = scaler.transform(XDev)
    XVal = scaler.transform(XVal)
    XTest = scaler.transform(XTest)
    #scalerfile = 'scaler_'+train_DM+'.sav'
    #joblib.dump(scaler, scalerfile)

    # Linear dimensionality reduction using "Singular Value Decomposition" of the data to project it to a lower dimensional space.
    # The input data is centered but not scaled for each feature before applying the SVD.
    '''
    pca = decomposition.PCA(n_components=len(trainFeatures)).fit(XDev)
    XDev = pca.transform(XDev)
    XVal = pca.transform(XVal)
    XTest = pca.transform(XTest)
    '''
    f.write("Preparing DATA took:   {0:.2f}s\n".format(time.time() - start))
    f.close()

    print '  Development (train):', len(dataDev)#, '(', dataDev.EventWeight.sum()*luminosity, 'weighted)'
    print '    Signal:', len(dataDev[dataDev.category == 0])#, '(', dataDev[dataDev.category == 1].EventWeight.sum()*luminosity, 'weighted)'
    print '    Background1:', len(dataDev[dataDev.category == 1])#, '(', dataDev[dataDev.category == 0].EventWeight.sum()*luminosity, 'weighted)'
    print '    Background2:', len(dataDev[dataDev.category == 2])
    print '    Background3:', len(dataDev[dataDev.category == 3])
    print '    Background4:', len(dataDev[dataDev.category == 4])
    print '    Background5:', len(dataDev[dataDev.category == 5])
    print '  Validation:', len(dataVal)#, '(', dataVal.EventWeight.sum()*luminosity, 'weighted)'
    print '    Signal:', len(dataVal[dataVal.category == 0])#, '(', dataVal[dataVal.category == 1].EventWeight.sum()*luminosity, 'weighted)'
    print '    Background1:', len(dataVal[dataVal.category == 1])#, '(', dataVal[dataVal.category == 0].EventWeight.sum()*luminosity, 'weighted)'
    print '    Background2:', len(dataVal[dataVal.category == 2])
    print '    Background3:', len(dataVal[dataVal.category == 3])
    print '    Background4:', len(dataVal[dataVal.category == 4])
    print '    Background5:', len(dataVal[dataVal.category == 5])
    print '  Test:', len(dataTest)#, '(', dataTest.EventWeight.sum()*luminosity, 'weighted)'
    print '    Signal:', len(dataTest[dataTest.category == 0])#, '(', dataTest[dataTest.category == 1].EventWeight.sum()*luminosity, 'weighted)'
    print '    Background1:', len(dataTest[dataTest.category == 1])#, '(', dataTest[dataTest.category == 0].EventWeight.sum()*luminosity, 'weighted)'
    print '    Background2:', len(dataTest[dataTest.category == 2])
    print '    Background3:', len(dataTest[dataTest.category == 3])
    print '    Background4:', len(dataTest[dataTest.category == 4])
    print '    Background5:', len(dataTest[dataTest.category == 5])
    print "DATA is ready!"
    print("Preparing DATA took: {0:.2f}s".format(time.time() - start))

    return dataDev, dataVal, dataTest, XDev, YDev, weightDev, XVal, YVal, weightVal, XTest, YTest, weightTest

filepath = cfg.lgbk
name = 'Model_Ver_1'
fraction = 0.05
dataLoader(filepath, name, fraction)
