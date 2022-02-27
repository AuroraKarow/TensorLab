LAYER_BEGIN

struct Layer
{
    uint64_t iActFuncType = NULL, iLayerType = FC;
    double dLayerLearnRate = 0;

    virtual void ValueAssign(Layer &lyrSrc)
    {
        iLayerType = lyrSrc.iLayerType;
        iActFuncType = lyrSrc.iActFuncType;
        dLayerLearnRate = lyrSrc.dLayerLearnRate;
    }
    virtual void ValueCopy(Layer &lyrSrc) { ValueAssign(lyrSrc); }
    virtual void ValueMove(Layer &&lyrSrc) {}
    Layer(Layer &lyrSrc) { ValueCopy(lyrSrc); }
    void operator=(Layer &lyrSrc) { ValueCopy(lyrSrc); }
    
    Layer(uint64_t iLayerTypeVal = FC, uint64_t iActFuncTypeVal = NULL, double dLearnRate = 0) : iLayerType(iLayerTypeVal), iActFuncType(iActFuncTypeVal), dLayerLearnRate(dLearnRate) {}
    template<typename T, typename = std::enable_if_t<true, vect>, typename = std::enable_if_t<true, feature>> set<T> Activate(set<T> &setInput)
    {
        switch (iActFuncType)
        {
        case SIGMOID:   return activate(setInput, sigmoid);
        case RELU:      return activate(setInput, ReLU);
        case SOFTMAX:   return activate(setInput, softmax);
        default: return std::move(setInput);
        }
    }
    // In case softmax - setActInput -> Softmax output; setGrad -> Label origin
    template<typename T, typename = std::enable_if_t<true, vect>, typename = std::enable_if_t<true, feature>> set<T> Derivative(set<T> &setActInput, set<T> &setGrad)
    {
        switch (iActFuncType)
        {
        case SIGMOID:   return derivative(setActInput, setGrad, sigmoid_dv);
        case RELU:      return derivative(setActInput, setGrad, ReLU_dv);
        case SOFTMAX:   return softmax_cec_grad(setActInput, setGrad);
        default: return std::move(setGrad);
        }
    }
    void ForwProp() {}
    void BackProp() {}

    virtual void Reset() {}
    ~Layer() {}
};

struct LayerFC : Layer
{
    vect vecLayerWeight;
    set<vect> setLayerInput, setLayerOutput;
    // Default
    _ADA AdaDeltaVect advLayerDelta;

    void ValueAssign(LayerFC &lyrSrc) {}
    void ValueCopy(LayerFC &lyrSrc)
    {
        vecLayerWeight = lyrSrc.vecLayerWeight;
        setLayerInput = lyrSrc.setLayerInput;
        setLayerOutput = lyrSrc.setLayerOutput;
    }
    void ValueMove(LayerFC &&lyrSrc)
    {
        vecLayerWeight = std::move(lyrSrc.vecLayerWeight);
        setLayerInput = std::move(lyrSrc.setLayerInput);
        setLayerOutput = std::move(lyrSrc.setLayerOutput);
    }
    LayerFC(LayerFC &lyrSrc) : Layer(lyrSrc) { ValueCopy(lyrSrc); }
    LayerFC(LayerFC &&lyrSrc) : Layer(lyrSrc) { ValueMove(std::move(lyrSrc)); }
    void operator=(LayerFC &lyrSrc) { Layer::operator=(lyrSrc); ValueCopy(lyrSrc); }
    void operator=(LayerFC &&lyrSrc) { Layer::operator=(std::move(lyrSrc)); ValueMove(std::move(lyrSrc)); }
    
    LayerFC() : Layer(FC) {}
    LayerFC(uint64_t iInputLnCnt, uint64_t iOutputLnCnt, uint64_t iActFuncTypeVal = SIGMOID, double dLearnRate = 0, double dRandBoundryFirst = 0, double dRandBoundrySecond = 0, double dAcc = 1e-05) : Layer(FC, iActFuncTypeVal, dLearnRate) { vecLayerWeight = _FC InitWeight(iInputLnCnt, iOutputLnCnt, dRandBoundryFirst, dRandBoundrySecond, dAcc); }
    set<vect> ForwProp(set<vect> &setInput, bool bFirstLayer = false)
    {
        if(bFirstLayer) { if(!setLayerInput.size()) setLayerInput = setInput; }
        else setLayerInput = std::move(setInput);
        setLayerOutput = _FC Output(setLayerInput, vecLayerWeight);
        return Activate(setLayerOutput);
    }
    set<vect> BackProp(set<vect> &setGradOrOutput, set<vect> &setOrigin = blank_vect_seq)
    {
        if(setOrigin.size()) setGradOrOutput = Derivative(setGradOrOutput, setOrigin);
        else setGradOrOutput = Derivative(setLayerOutput, setGradOrOutput);
        auto setGradBack = _FC GradLossToInput(setGradOrOutput, vecLayerWeight);
        auto vecGradWeight = _FC GradLossToWeight(setGradOrOutput, setLayerInput);
        if(dLayerLearnRate) vecLayerWeight -= dLayerLearnRate * vecGradWeight;
        else vecLayerWeight = _FC AdaDeltaUpdateWeight(vecLayerWeight, vecGradWeight, advLayerDelta);
        return setGradBack;
    }

    void Reset()
    {
        vecLayerWeight.reset();
        setLayerInput.reset();
        setLayerOutput.reset();
        advLayerDelta.reset();
    }
    ~LayerFC() { Reset(); }
};

struct LayerFCBN : Layer
{
    // Shift, Scale, Dominant
    double dBeta = 0, dGamma = 1, dEpsilon = 1e-10;
    set<vect> setLayerInput;
    _ADA AdaDeltaVal advBeta, advGamma;
    BN_FC BNData;

    void ValueAssign(LayerFCBN &lyrSrc)
    {
        dBeta = lyrSrc.dBeta;
        dGamma = lyrSrc.dGamma;
        dEpsilon = lyrSrc.dEpsilon;
    }
    void ValueCopy(LayerFCBN &lyrSrc)
    {
        ValueAssign(lyrSrc);
        setLayerInput = lyrSrc.setLayerInput;
        advBeta = lyrSrc.advBeta;
        advGamma = lyrSrc.advGamma;
        BNData = lyrSrc.BNData;
    }
    void ValueMove(LayerFCBN &&lyrSrc)
    {
        ValueAssign(lyrSrc);
        setLayerInput = std::move(lyrSrc.setLayerInput);
        advBeta = std::move(lyrSrc.advBeta);
        advGamma = std::move(lyrSrc.advGamma);
        BNData = std::move(lyrSrc.BNData);
    }
    LayerFCBN(LayerFCBN &lyrSrc) : Layer(lyrSrc) { ValueCopy(lyrSrc); }
    LayerFCBN(LayerFCBN &&lyrSrc) : Layer(lyrSrc) { ValueMove(std::move(lyrSrc)); }
    void operator=(LayerFCBN &lyrSrc) { Layer::operator=(lyrSrc); ValueCopy(lyrSrc); }
    void operator=(LayerFCBN &&lyrSrc) { Layer::operator=(std::move(lyrSrc)); ValueMove(std::move(lyrSrc)); }

    LayerFCBN(double dShift = 0, double dScale = 1, uint64_t iActFuncTypeVal = SIGMOID, double dLearnRate = 0, double dDmt = 1e-10) : Layer(FC_BN, iActFuncTypeVal, dLearnRate), dBeta(dShift), dGamma(dScale), dEpsilon(dDmt) {}
    set<vect> ForwProp(set<vect> &setInput, bool bFirstLayer = false)
    {
        if(bFirstLayer) { if(!setLayerInput.size()) setLayerInput = setInput; }
        else setLayerInput = std::move(setInput);
        BNData = _FC BNTrain(setLayerInput, dBeta, dGamma, dEpsilon);
        return Activate(BNData.setY);
    }
    set<vect> BackProp(set<vect> &setGradOrOutput, set<vect> &setOrigin = blank_vect_seq)
    {
        if(setOrigin.size()) setGradOrOutput = Derivative(setGradOrOutput, setOrigin); 
        else setGradOrOutput = Derivative(BNData.setY, setGradOrOutput);
        auto setGradBack = _FC BNGradLossToInput(BNData, setLayerInput, setGradOrOutput, dGamma, dEpsilon);
        auto dGradScale = _FC BNGradLossToScale(setGradOrOutput, BNData),
            dGradShift = _FC BNGradLossToShift(setGradOrOutput);
        if(dLayerLearnRate)
        {
            dGamma -= dLayerLearnRate * dGradScale;
            dBeta -= dLayerLearnRate * dGradShift;
        }
        else
        {
            dGamma = _FC BNAdaDeltaUpdateScaleShift(dGamma, dGradScale, advGamma);
            dBeta = _FC BNAdaDeltaUpdateScaleShift(dBeta, dGradShift, advBeta);
        }
        return setGradBack;
    }

    void Reset() { setLayerInput.reset(); }
    ~LayerFCBN() { Reset(); }
};

struct LayerConv : Layer
{
    uint64_t iLayerLnStride = 0, iLayerColStride = 0, iLayerLnDilation = 0, iLayerColDilation = 0, iLayerInputPadTop = 0, iLayerInputPadRight = 0, iLayerInputPadBottom = 0, iLayerInputPadLeft = 0, iLayerLnDistance = 0, iLayerColDistance = 0;
    tensor tenKernel;
    set<feature> setLayerInput, setLayerOutput;
    _ADA ada_tensor<_ADA AdaDeltaVect> advLayerDelta;

    void ValueAssign(LayerConv &lyrSrc)
    {
        iLayerLnStride = lyrSrc.iLayerLnStride;
        iLayerColStride = lyrSrc.iLayerColStride;
        iLayerLnDilation = lyrSrc.iLayerLnDilation;
        iLayerColDilation = lyrSrc.iLayerColDilation;
        iLayerInputPadTop = lyrSrc.iLayerInputPadTop;
        iLayerInputPadRight = lyrSrc.iLayerInputPadRight;
        iLayerInputPadBottom = lyrSrc.iLayerInputPadBottom;
        iLayerInputPadLeft = lyrSrc.iLayerInputPadLeft;
        iLayerLnDistance = lyrSrc.iLayerLnDistance;
        iLayerColDistance = lyrSrc.iLayerColDistance;
    }
    void ValueCopy(LayerConv &lyrSrc)
    {
        ValueAssign(lyrSrc);
        tenKernel = lyrSrc.tenKernel;
        setLayerInput = lyrSrc.setLayerInput;
        setLayerOutput = lyrSrc.setLayerOutput;
        advLayerDelta = lyrSrc.advLayerDelta;
    }
    void ValueMove(LayerConv &&lyrSrc)
    {
        ValueAssign(lyrSrc);
        tenKernel = std::move(lyrSrc.tenKernel);
        setLayerInput = std::move(lyrSrc.setLayerInput);
        setLayerOutput = std::move(lyrSrc.setLayerOutput);
        advLayerDelta = std::move(lyrSrc.advLayerDelta);
    }
    LayerConv(LayerConv &lyrSrc) : Layer(lyrSrc) { ValueCopy(lyrSrc); }
    LayerConv(LayerConv &&lyrSrc) : Layer(lyrSrc) { ValueMove(std::move(lyrSrc)); }
    void operator=(LayerConv &lyrSrc) { Layer::operator=(lyrSrc); ValueCopy(lyrSrc); }
    void operator=(LayerConv &&lyrSrc) { Layer::operator=(std::move(lyrSrc)); ValueMove(std::move(lyrSrc)); }

    LayerConv() : Layer(CONV) {}
    LayerConv(uint64_t iKernelAmt, uint64_t iKernelChannCnt, uint64_t iKernelLnCnt, uint64_t iKernelColCnt, uint64_t iLnStride, uint64_t iColStride, uint64_t iActFuncTypeVal = RELU, double dLearnRate = 0, double dRandBoundryFirst = 0, double dRandBoundrySecond = 0, double dRandBoundryAcc = 1e-5, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0) : Layer(CONV, iActFuncTypeVal, dLearnRate), iLayerLnStride(iLnStride), iLayerColStride(iColStride), iLayerLnDilation(iLnDilation), iLayerColDilation(iColDilation), iLayerInputPadTop(iInputPadTop), iLayerInputPadRight(iInputPadRight), iLayerInputPadBottom( iInputPadBottom), iLayerInputPadLeft(iInputPadLeft), iLayerLnDistance(iLnDistance), iLayerColDistance(iColDistance) { tenKernel = _CONV InitKernel(iKernelAmt, iKernelChannCnt, iKernelLnCnt, iKernelColCnt, dRandBoundryFirst, dRandBoundrySecond, dRandBoundryAcc); }
    set<feature> ForwProp(set<feature> &setInput, bool bFirstLayer = false)
    {
        if(bFirstLayer) { if(!setLayerInput.size()) setLayerInput = setInput; }
        else setLayerInput = std::move(setInput);
        setLayerOutput = _CONV Conv(setLayerInput, tenKernel, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance);
        return Activate(setLayerOutput);
    }
    set<feature> BackProp(set<feature> &setGrad)
    {
        setGrad = Derivative(setLayerOutput, setGrad);
        auto setGradBack = _CONV GradLossToInput(setGrad, tenKernel, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance);
        auto tenGradKernel = _CONV GradLossToKernel(setGrad, setLayerInput, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance);
        if(dLayerLearnRate) tenKernel = _CONV UpdateKernel(tenKernel, tenGradKernel, dLayerLearnRate);
        else tenKernel = _CONV AdaDeltaUpdateKernel(tenKernel, tenGradKernel, advLayerDelta);
        return setGradBack;
    }

    void Reset()
    {
        setLayerInput.reset();
        setLayerOutput.reset();
        advLayerDelta.reset();
    }
    ~LayerConv() { Reset(); }
};

struct LayerConvBN : Layer
{
    // Dominant
    double dEpsilon = 1e-5;
    // Shift, Scale
    vect vecBeta, vecGamma;
    set<feature> setLayerInput;
    _ADA AdaDeltaVect advBeta, advGamma;
    BN_CONV BNData;

    void ValueAssign(LayerConvBN &lyrSrc) { dEpsilon = lyrSrc.dEpsilon;}
    void ValueCopy(LayerConvBN &lyrSrc)
    {
        ValueAssign(lyrSrc);
        vecBeta = lyrSrc.vecBeta;
        vecGamma = lyrSrc.vecGamma;
        setLayerInput = lyrSrc.setLayerInput;
        advBeta = lyrSrc.advBeta;
        advGamma = lyrSrc.advGamma;
        BNData = lyrSrc.BNData;
    }
    void ValueMove(LayerConvBN &&lyrSrc)
    {
        ValueAssign(lyrSrc);
        vecBeta = std::move(lyrSrc.vecBeta);
        vecGamma = std::move(lyrSrc.vecGamma);
        setLayerInput = std::move(lyrSrc.setLayerInput);
        advBeta = std::move(lyrSrc.advBeta);
        advGamma = std::move(lyrSrc.advGamma);
        BNData = std::move(lyrSrc.BNData);
    }
    LayerConvBN(LayerConvBN &lyrSrc) : Layer(lyrSrc) { ValueCopy(lyrSrc); }
    LayerConvBN(LayerConvBN &&lyrSrc) : Layer(lyrSrc) { ValueMove(std::move(lyrSrc)); }
    void operator=(LayerConvBN &lyrSrc) { Layer::operator=(lyrSrc); ValueCopy(lyrSrc); }
    void operator=(LayerConvBN &&lyrSrc) { Layer::operator=(std::move(lyrSrc)); ValueMove(std::move(lyrSrc)); }

    LayerConvBN(uint64_t iChannCnt = 1, double dShift = 0, double dScale = 1, uint64_t iActFuncTypeVal = RELU, double dLearnRate = 0, double dDmt = 1e-10) : Layer(CONV_BN, iActFuncTypeVal, dLearnRate), dEpsilon(dDmt)
    {
        vecBeta = _CONV BNInitScaleShift(iChannCnt, dShift);
        vecGamma = _CONV BNInitScaleShift(iChannCnt, dScale);
    }
    set<feature> ForwProp(set<feature> &setInput, bool bFirstLayer = false)
    {
        if(bFirstLayer) { if(!setLayerInput.size()) setLayerInput = setInput; }
        else setLayerInput = std::move(setInput);
        BNData = _CONV BNTrain(setLayerInput, vecBeta, vecGamma, dEpsilon);
        return Activate(BNData.setY);
    }
    set<feature> BackProp(set<feature> &setGrad)
    {
        setGrad = Derivative(BNData.setY, setGrad);
        auto setGradBack = _CONV BNGradLossToInput(BNData, setLayerInput, setGrad, vecGamma, dEpsilon);
        auto dGradScale = _CONV BNGradLossToScale(setGrad, BNData),
            dGradShift = _CONV BNGradLossToShift(setGrad);
        if(dLayerLearnRate)
        {
            vecGamma -= dLayerLearnRate * dGradScale;
            vecBeta -= dLayerLearnRate * dGradShift;
        }
        else
        {
            vecGamma = _CONV BNAdaDeltaUpdateScaleShift(vecGamma, dGradScale, advGamma);
            vecBeta = _CONV BNAdaDeltaUpdateScaleShift(vecBeta, dGradShift, advBeta);
        }
        return setGradBack;
    }

    void Reset()
    {
        vecBeta.reset();
        vecGamma.reset();
        setLayerInput.reset();
        advBeta.reset();
        advGamma.reset();
    }
    ~LayerConvBN() { Reset(); }
};

struct LayerPool : Layer
{
    uint64_t iPoolType = POOL_MAX, iLayerFilterLnCnt = 0, iLayerFilterColCnt = 0, iLayerLnStride = 0, iLayerColStride = 0, iLayerLnDilation = 0, iLayerColDilation = 0;
    set<feature> setLayerInput, setLayerOutput;

    void ValueAssign(LayerPool &lyrSrc)
    {
        iPoolType = lyrSrc.iPoolType;
        iLayerLnStride = lyrSrc.iLayerLnStride;
        iLayerColStride = lyrSrc.iLayerColStride;
        iLayerLnDilation = lyrSrc.iLayerLnDilation;
        iLayerColDilation = lyrSrc.iLayerColDilation;
        iLayerFilterLnCnt = lyrSrc.iLayerFilterLnCnt;
        iLayerFilterColCnt = lyrSrc.iLayerFilterColCnt;
    }
    void ValueCopy(LayerPool &lyrSrc)
    {
        ValueAssign(lyrSrc);
        setLayerInput = lyrSrc.setLayerInput;
        setLayerOutput = lyrSrc.setLayerOutput;
    }
    void ValueMove(LayerPool &&lyrSrc)
    {
        ValueAssign(lyrSrc);
        setLayerInput = std::move(lyrSrc.setLayerInput);
        setLayerOutput = std::move(lyrSrc.setLayerOutput);
    }
    LayerPool(LayerPool &lyrSrc) : Layer(lyrSrc) { ValueCopy(lyrSrc); }
    LayerPool(LayerPool &&lyrSrc) : Layer(lyrSrc) { ValueMove(std::move(lyrSrc)); }
    void operator=(LayerPool &lyrSrc) { Layer::operator=(lyrSrc); ValueCopy(lyrSrc); }
    void operator=(LayerPool &&lyrSrc) { Layer::operator=(std::move(lyrSrc)); ValueMove(std::move(lyrSrc)); }

    uint64_t PoolUpType(uint64_t iPoolIdx)
    {
        switch (iPoolIdx)
        {
        case POOL_AVG: return POOL_UP_AVG;
        case POOL_GAG: return POOL_UP_GAG;
        case POOL_MAX: return POOL_UP_MAX;
        default: return (uint64_t)NAN;
        }
    }
    LayerPool(uint64_t iPoolTypeVal = POOL_MAX, uint64_t iFilterLnCnt = 0, uint64_t iFilterColCnt = 0, uint64_t iLnStride = 0, uint64_t iColStride = 0, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iActFuncTypeVal = NULL) : Layer(POOL, iActFuncTypeVal, 0), iPoolType(iPoolTypeVal), iLayerFilterLnCnt(iFilterLnCnt), iLayerFilterColCnt(iFilterColCnt), iLayerLnStride(iLnStride), iLayerColStride(iColStride), iLayerLnDilation(iLnDilation), iLayerColDilation(iColDilation) {}
    set<feature> ForwProp(set<feature> &setInput, bool bFirstLayer = false)
    {
        if(bFirstLayer) { if(!setLayerInput.size()) setLayerInput = setInput; }
        else setLayerInput = std::move(setInput);
        setLayerOutput = _CONV Pool(setLayerInput, iPoolType, true, set<feature>(), iLayerFilterLnCnt, iLayerFilterColCnt, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation);
        return Activate(setLayerOutput);
    }
    set<feature> BackProp(set<feature> &setGrad) { return _CONV Pool(Derivative(setLayerOutput, setGrad), PoolUpType(iPoolType), false, setLayerInput, iLayerFilterLnCnt, iLayerFilterColCnt, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation); }

    void Reset()
    {
        setLayerInput.reset();
        setLayerOutput.reset();
    }
    ~LayerPool() { Reset(); }
};

struct LayerTrans : Layer
{
    uint64_t iLnCnt = 0, iColCnt = 0;
    bool bFeatToVec = true;

    void ValueAssign(LayerTrans &lyrSrc)
    {
        iLnCnt = lyrSrc.iLnCnt;
        iColCnt = lyrSrc.iColCnt;
        bFeatToVec = lyrSrc.bFeatToVec;
    }
    void ValueCopy(LayerTrans &lyrSrc) { ValueAssign(lyrSrc); }
    void ValueMove(LayerTrans &lyrSrc) {}
    LayerTrans(LayerTrans &lyrSrc) : Layer(lyrSrc) { ValueCopy(lyrSrc); }
    void operator=(LayerTrans &lyrSrc) { Layer::operator=(lyrSrc); ValueCopy(lyrSrc); }

    LayerTrans(uint64_t iActFuncTypeVal = NULL) : Layer(TRANS, iActFuncTypeVal, 0) {}
    set<vect> ForwProp(set<feature> &setInput)
    {
        iLnCnt = setInput[IDX_ZERO][IDX_ZERO].LN_CNT;
        iColCnt = setInput[IDX_ZERO][IDX_ZERO].COL_CNT;
        return _FC FeatureTransform(setInput);
    }
    set<feature> BackProp(set<vect> &setGrad) {return _FC FeatureTransform(setGrad, iLnCnt, iColCnt);}
    
    LayerTrans(uint64_t iChannLnCnt, uint64_t iChannColCnt, uint64_t iActFuncIdx = NULL) : Layer(TRANS, iActFuncIdx, 0), iLnCnt(iChannLnCnt), iColCnt(iChannColCnt), bFeatToVec(false) {}
    set<feature> ForwProp(set<vect> &setInput) {return _FC FeatureTransform(setInput, iLnCnt, iColCnt);}
    set<vect> BackProp(set<feature> &setGrad) {return _FC FeatureTransform(setGrad);}

    void Reset() {}
    ~LayerTrans() {}
};

LAYER_END