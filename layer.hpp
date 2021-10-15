LAYER_BEGIN

class Layer
{
protected:
    uint64_t iActFuncType = NULL_FUNC;
    bool bFirstLayer = false;
    template<typename T, typename = std::enable_if_t<true, vect>, typename = std::enable_if_t<true, feature>> set<T> Activate(set<T> &setInput)
    {
        switch (iActFuncType)
        {
        case SIGMOID:   return activate(setInput, sigmoid);
        case RELU:      return activate(setInput, ReLU);
        case SOFTMAX:   return activate(setInput, softmax);  
        default: return setInput;
        }
    }
    template<typename T, typename = std::enable_if_t<true, vect>, typename = std::enable_if_t<true, feature>> set<T> Derivative(set<T> &setActInput, set<T> &setGrad)
    {
        switch (iActFuncType)
        {
        case SIGMOID:   return derivative(setActInput, setGrad, sigmoid_dv);
        case RELU:      return derivative(setActInput, setGrad, ReLU_dv);
        default: return setGrad;
        }
    }
public:
    uint64_t iLayerType = 0;
    virtual void setActFunc(uint64_t iActFuncIdx) { iActFuncType = iActFuncIdx;}
    Layer(uint64_t iActFuncIdx = SIGMOID, uint64_t iCurrLayerType = 0, bool bIsFirstLayer = false) : iLayerType(iCurrLayerType), iActFuncType(iActFuncIdx), bFirstLayer(bIsFirstLayer) {}
    void operator=(Layer &lyrSrc) {new (this)Layer(lyrSrc);}
    Layer(Layer &lyrSrc)
    {
        iActFuncType = lyrSrc.iActFuncType;
        bFirstLayer = lyrSrc.bFirstLayer;
        iLayerType = lyrSrc.iLayerType;
    }
    void ForwProp() {}
    void BackProp() {}
    void InitPara() {}
};

class LayerFC : public Layer
{
protected:
    vect vecLayerWeight;
    set<vect> setLayerInput;
    set<vect> setOutput;
public:
    bool InitLayerWeight(uint64_t iInputLnCnt, uint64_t iOutputLnCnt, double dRandBoundryFirst = 0.0, double dRandBoundrySecond = 0.0, double dAcc = 1e-05)
    {
        if(iInputLnCnt && iOutputLnCnt)
        {
            vecLayerWeight = _FC InitWeight(iInputLnCnt, iOutputLnCnt, dRandBoundryFirst, dRandBoundrySecond, dAcc);
            return true;
        }
        else return false;
    }
    LayerFC(uint64_t iInputLnCnt, uint64_t iOutputLnCnt, uint64_t iActFuncIdx = SIGMOID, uint64_t iCurrLayerType = FC, bool bIsFirstLayer = false, double dRandBoundryFirst = 0.0, double dRandBoundrySecond = 0.0, double dAcc = 1e-05) : Layer(iActFuncIdx, iCurrLayerType, bIsFirstLayer) {InitLayerWeight(iInputLnCnt, iOutputLnCnt, dRandBoundryFirst, dRandBoundrySecond, dAcc);}
    set<vect> ForwProp(set<vect> &setInput)
    {
        if(bFirstLayer)setLayerInput = setInput;
        else setLayerInput = std::move(setInput);
        setOutput = _FC Output(setLayerInput, vecLayerWeight);
        return Activate(setOutput);
    }
    set<vect> BackProp(set<vect> &setGrad, double dLearnRate)
    {
        setGrad = Derivative(setOutput, setGrad);
        auto setGradBack = _FC GradLossToInput(setGrad, vecLayerWeight);
        vecLayerWeight -= dLearnRate * _FC GradLossToWeight(setGrad, setLayerInput);
        return setGradBack;
    }
    LayerFC(LayerFC &lyrSrc) : Layer(lyrSrc)
    {
        vecLayerWeight = lyrSrc.vecLayerWeight;
        setLayerInput = lyrSrc.setLayerInput;
    }
    LayerFC(LayerFC &&lyrSrc) : Layer(lyrSrc)
    {
        vecLayerWeight = std::move(lyrSrc.vecLayerWeight);
        setLayerInput = std::move(lyrSrc.setLayerInput);
    }
    void operator=(LayerFC &lyrSrc) {new(this)LayerFC(lyrSrc);}
    void operator=(LayerFC &&lyrSrc) {new(this)LayerFC(std::move(lyrSrc));}
};

class LayerFCAda : public LayerFC
{
protected:
    _ADA AdaDeltaVect advLayerDelta;
public:
    LayerFCAda(uint64_t iInputLnCnt, uint64_t iOutputLnCnt, uint64_t iActFuncIdx = SIGMOID, double dDecayController = 0.95, double dAdaDominator = 1e-6, uint64_t iCurrLayerType = FC_ADA, bool bIsFirstLayer = false, double dRandBoundryFirst = 0.0, double dRandBoundrySecond = 0.0, double dAcc = 1e-05) : LayerFC(iInputLnCnt, iOutputLnCnt, iActFuncIdx, iCurrLayerType, bIsFirstLayer, dRandBoundryFirst, dRandBoundrySecond, dAcc) {advLayerDelta = _ADA AdaDeltaVect(vecLayerWeight.LN_CNT, vecLayerWeight.COL_CNT, dDecayController, dAdaDominator);}
    set<vect> BackProp(set<vect> &setGrad)
    {
        setGrad = Derivative(setOutput, setGrad);
        auto setGradBack = _FC GradLossToInput(setGrad, vecLayerWeight);
        vecLayerWeight = _FC AdaDeltaUpdateWeight(vecLayerWeight, _FC GradLossToWeight(setGrad, setLayerInput), advLayerDelta);
        return setGradBack;
    }
    LayerFCAda(LayerFCAda &lyrSrc) : LayerFC(lyrSrc) {advLayerDelta = lyrSrc.advLayerDelta;}
    LayerFCAda(LayerFCAda &&lyrSrc) : LayerFC(std::move(lyrSrc)) {advLayerDelta = std::move(lyrSrc.advLayerDelta);}
    void operator=(LayerFCAda &lyrSrc) {new(this)LayerFCAda(lyrSrc);}
    void operator=(LayerFCAda &&lyrSrc) {new(this)LayerFCAda(std::move(lyrSrc));}
};

class LayerBNFC : public Layer
{
protected:
    _FC FCBN fcbnData;
    // Shift
    double dBeta = 0;
    // Scale
    double dGamma = 1;
    double dEpsilon = 1e-10;
    set<vect> setLayerInput;
    void ValueAssign(LayerBNFC &lyrSrc)
    {
        dBeta = lyrSrc.dBeta;
        dGamma = lyrSrc.dGamma;
    }
public:
    LayerBNFC(uint64_t iActFuncIdx = NULL_FUNC, double dShift = 0, double dScale = 1, uint64_t iCurrLayerType = BN_FC, bool bIsFirstLayer = false, double dBNDominator = 1e-10) : Layer(iActFuncIdx, iCurrLayerType, bIsFirstLayer), dBeta(dShift), dGamma(dScale), dEpsilon(dBNDominator) {}
    set<vect> ForwProp(set<vect> &setInput)
    {
        if(bFirstLayer) setLayerInput = setInput;
        else setLayerInput = std::move(setInput);
        fcbnData = _FC BNTrain(setLayerInput, dBeta, dGamma, true, dEpsilon);
        return Activate(fcbnData.setY);
    }
    set<vect> BackProp(set<vect> &setGrad, double dLearnRate)
    { 
        setGrad = Derivative(fcbnData.setY, setGrad);
        auto setGradBack = _FC BNGradLossToInput(fcbnData, setLayerInput, setGrad, dGamma, dEpsilon);
        dGamma = _FC BNUpdateScaleShift(dGamma, _FC BNGradLossToScale(setGrad, fcbnData), dLearnRate);
        dBeta = _FC BNUpdateScaleShift(dBeta, _FC BNGradLossToShift(setGrad), dLearnRate);
        return setGradBack;
    }
    set<vect> Deduce(uint64_t iMiniBatchSize) {return _FC BNDeduce(setLayerInput, dBeta, dGamma, iMiniBatchSize, dEpsilon);}
    LayerBNFC(LayerBNFC &lyrSrc) : Layer(lyrSrc)
    {
        fcbnData = lyrSrc.fcbnData;
        ValueAssign(lyrSrc);
        setLayerInput = lyrSrc.setLayerInput;
    }
    LayerBNFC(LayerBNFC &&lyrSrc) : Layer(lyrSrc)
    {
        fcbnData = std::move(lyrSrc.fcbnData);
        ValueAssign(lyrSrc);
        setLayerInput = std::move(lyrSrc.setLayerInput);
    }
    void operator=(LayerBNFC &lyrSrc) {new(this)LayerBNFC(lyrSrc);}
    void operator=(LayerBNFC &&lyrSrc){new(this)LayerBNFC(std::move(lyrSrc));}
};

class LayerBNFCAda : public LayerBNFC
{
protected: 
    _ADA AdaDeltaVal advBeta;
    _ADA AdaDeltaVal advGamma;
public:
    LayerBNFCAda(uint64_t iActFuncIdx = NULL_FUNC, double dShift = 0, double dScale = 1, uint64_t iCurrLayerType = BN_FC_ADA, bool bIsFirstLayer = false, double dDecayController = 0.95, double dAdaDominator = 1e-6, double dBNDominator = 1e-10) : LayerBNFC(iActFuncIdx, dShift, dScale, iCurrLayerType, bIsFirstLayer, dBNDominator)
    {
        advBeta = _ADA AdaDeltaVal(dDecayController, dAdaDominator);
        advGamma = _ADA AdaDeltaVal(dDecayController, dAdaDominator);
    }
    set<vect> BackProp(set<vect> &setGrad)
    {
        setGrad = Derivative(fcbnData.setY, setGrad);
        auto setGradBack = _FC BNGradLossToInput(fcbnData, setLayerInput, setGrad, dGamma, dEpsilon);
        dGamma = _FC BNAdaDeltaUpdateScaleShift(dGamma, _FC BNGradLossToScale(setGrad, fcbnData), advGamma);
        dBeta = _FC BNAdaDeltaUpdateScaleShift(dBeta, _FC BNGradLossToShift(setGrad), advBeta);
        return setGradBack;
    }
    LayerBNFCAda(LayerBNFCAda &lyrSrc) : LayerBNFC(lyrSrc)
    {
        advBeta = lyrSrc.advBeta;
        advGamma = lyrSrc.advGamma;
    }
    void operator=(LayerBNFCAda &lyrSrc) {new(this)LayerBNFCAda(lyrSrc);}
};

class LayerConv : public Layer
{
protected:
    tensor tenKernel;
    set<feature> setLayerInput;
    set<feature> setOutput;
    uint64_t iLayerLnStride = 0;
    uint64_t iLayerColStride = 0;
    uint64_t iLayerLnDilation = 0;
    uint64_t iLayerColDilation = 0;
    uint64_t iLayerInputPadTop = 0;
    uint64_t iLayerInputPadRight = 0;
    uint64_t iLayerInputPadBottom = 0;
    uint64_t iLayerInputPadLeft = 0;
    uint64_t iLayerLnDistance = 0;
    uint64_t iLayerColDistance = 0;
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
public:
    LayerConv(LayerConv &lyrSrc) : Layer(lyrSrc)
    {
        tenKernel = lyrSrc.tenKernel;
        setLayerInput = lyrSrc.setLayerInput;
        ValueAssign(lyrSrc);
    }
    LayerConv(LayerConv &&lyrSrc) : Layer(lyrSrc)
    {
        tenKernel = std::move(lyrSrc.tenKernel);
        ValueAssign(lyrSrc);
        setLayerInput = std::move(lyrSrc.setLayerInput);
    }
    void operator=(LayerConv &lyrSrc) {new(this)LayerConv(lyrSrc);}
    void operator=(LayerConv &&lyrSrc) {new(this)LayerConv(std::move(lyrSrc));}
    bool InitLayerKernel(uint64_t iAmt, uint64_t iChannCnt, uint64_t iLnCnt, uint64_t iColCnt, double dRandBoundryFirst = 0, double dRandBoundrySecond = 0, double dAcc = 1e-5)
    {
        if(iAmt && iChannCnt && iLnCnt && iColCnt)
        {
            tenKernel = _CONV InitKernel(iAmt, iChannCnt, iLnCnt, iColCnt, dRandBoundryFirst, dRandBoundrySecond, dAcc);
            return true;
        }
        else return false;
    }
    bool InitPara(uint64_t iLnStride, uint64_t iColStride, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0)
    {
        if(iLnStride && iColStride)
        {
            iLayerLnStride = iLnStride; iLayerColStride = iColStride;
            iLayerLnDilation = iLnDilation; iLayerColDilation = iColDilation;
            iLayerInputPadTop = iInputPadTop; iLayerInputPadRight = iInputPadRight; iLayerInputPadBottom = iInputPadBottom; iLayerInputPadLeft = iInputPadLeft;
            iLayerLnDistance = iLnDistance; iLayerColDistance = iColDistance;
            return true;
        }
        else return false;
    }
    LayerConv(uint64_t iKernelAmt, uint64_t iKernelChannCnt, uint64_t iKernelLnCnt, uint64_t iKernelColCnt, uint64_t iLnStride, uint64_t iColStride, uint64_t iActFuncIdx = RELU, uint64_t iCurrLayerType = CONV, bool bIsFirstLayer = false, double dRandBoundryFirst = 0, double dRandBoundrySecond = 0, double dAcc = 1e-5, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0) : Layer(iActFuncIdx, iCurrLayerType, bIsFirstLayer)
    {
        InitLayerKernel(iKernelAmt, iKernelChannCnt, iKernelLnCnt, iKernelColCnt, dRandBoundryFirst, dRandBoundrySecond, dAcc);
        InitPara(iLnStride, iColStride, iLnDilation, iColDilation, iInputPadTop, iInputPadRight, iInputPadBottom, iInputPadLeft, iLnDistance, iColDistance);
    }
    set<feature> ForwProp(set<feature> &setInput)
    {
        if(bFirstLayer) setLayerInput = setInput;
        else setLayerInput = std::move(setInput);
        setOutput = _CONV Conv(setLayerInput, tenKernel, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance);
        return Activate(setOutput);
    }
    set<feature> BackProp(set<feature> &setGrad, double dLearnRate) 
    {
        setGrad = Derivative(setOutput, setGrad);
        auto setGradBack = _CONV GradLossToInput(setGrad, tenKernel, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance);
        tenKernel = _CONV UpdateKernel(tenKernel, _CONV GradLossToKernel(setGrad, setLayerInput, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance), dLearnRate);
        return setGradBack;
    }
};

class LayerConvAda : public LayerConv
{
protected:
    _ADA ten_ada<_ADA AdaDeltaVect> advLayerDelta;
public:
    LayerConvAda(LayerConvAda &lyrSrc) : LayerConv(lyrSrc) {advLayerDelta = lyrSrc.advLayerDelta;}
    LayerConvAda(LayerConvAda &&lyrSrc) : LayerConv(std::move(lyrSrc)) {advLayerDelta = std::move(lyrSrc.advLayerDelta);}
    void operator=(LayerConvAda &lyrSrc) {new(this)LayerConvAda(lyrSrc);}
    void operator=(LayerConvAda &&lyrSrc) {new(this)LayerConvAda(std::move(lyrSrc));}
    LayerConvAda(uint64_t iKernelAmt, uint64_t iKernelChannCnt, uint64_t iKernelLnCnt, uint64_t iKernelColCnt, uint64_t iLnStride, uint64_t iColStride, uint64_t iActFuncIdx = RELU, uint64_t iCurrLayerType = CONV_ADA, bool bIsFirstLayer = false, double dDecayController = 0.95, double dRandBoundryFirst = 0, double dRandBoundrySecond = 0, double dAcc = 1e-5, double dAdaDominator = 1e-6, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0) : LayerConv(iKernelAmt, iKernelChannCnt, iKernelLnCnt, iKernelColCnt, iLnStride, iColStride, iActFuncIdx, iCurrLayerType, bIsFirstLayer, dRandBoundryFirst, dRandBoundrySecond, dAcc, iLnDilation, iColDilation, iInputPadTop, iInputPadRight, iInputPadBottom, iInputPadLeft, iLnDistance, iColDistance)
    {
        advLayerDelta.init(iKernelAmt);
        for(auto i=0; i<iKernelAmt; ++i)
        {
            advLayerDelta[i].init(iKernelChannCnt);
            for(auto j=0; j<iKernelChannCnt; ++j) advLayerDelta[i][j] = _ADA AdaDeltaVect(iKernelLnCnt, iKernelColCnt, dDecayController, dAdaDominator);
        }
    }
    set<feature> BackProp(set<feature> &setGrad)
    {
        setGrad = Derivative(setOutput, setGrad);
        auto setGradBack = _CONV GradLossToInput(setGrad, tenKernel, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance);
        tenKernel = _CONV AdaDeltaUpdateKernel(tenKernel, _CONV GradLossToKernel(setGrad, setLayerInput, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance), advLayerDelta);
        return setGradBack;
    }
};

class LayerPool : public Layer
{
protected:
    uint64_t iPoolType = POOL_MAX;
    set<feature> setLayerInput;
    set<feature> setOutput;
    uint64_t iLayerFilterLnCnt = 0;
    uint64_t iLayerFilterColCnt = 0;
    uint64_t iLayerLnStride = 0;
    uint64_t iLayerColStride = 0;
    uint64_t iLayerLnDilation = 0;
    uint64_t iLayerColDilation = 0;
    void ValueAssign(LayerPool &lyrSrc)
    {
        iLayerLnStride = lyrSrc.iLayerLnStride;
        iLayerColStride = lyrSrc.iLayerColStride;
        iLayerLnDilation = lyrSrc.iLayerLnDilation;
        iLayerColDilation = lyrSrc.iLayerColDilation;
        iLayerFilterLnCnt = lyrSrc.iLayerFilterLnCnt;
        iLayerFilterColCnt = lyrSrc.iLayerFilterColCnt;
    }
public:
    LayerPool(LayerPool &lyrSrc) : Layer(lyrSrc) 
    {
        setLayerInput = lyrSrc.setLayerInput;
        ValueAssign(lyrSrc);
    }
    LayerPool(LayerPool &&lyrSrc) : Layer(lyrSrc)
    {
        ValueAssign(lyrSrc);
        setLayerInput = std::move(lyrSrc.setLayerInput);
    }
    void operator=(LayerPool &lyrSrc) {new(this)LayerPool(lyrSrc);}
    void operator=(LayerPool &&lyrSrc) {new(this)LayerPool(std::move(lyrSrc));}
    static uint64_t PoolUpType(uint64_t iPoolIdx)
    {
        switch (iPoolIdx)
        {
        case POOL_AVG: return POOL_UP_AVG;
        case POOL_GAG: return POOL_UP_GAG;
        case POOL_MAX: return POOL_UP_MAX;
        default: return (uint64_t)NAN;
        }
    }
    void InitPara(uint64_t iFilterLnCnt = 0, uint64_t iFilterColCnt = 0, uint64_t iLnStride = 0, uint64_t iColStride = 0, uint64_t iLnDilation = 0, uint64_t iColDilation = 0)
    {
        iLayerFilterLnCnt = iFilterLnCnt; iLayerFilterColCnt = iFilterColCnt;
        iLayerLnStride = iLnStride; iLayerColStride = iColStride;
        iLayerLnDilation = iLnDilation; iLayerColDilation = iColDilation;
    }
    LayerPool(uint64_t iFilterLnCnt = 0, uint64_t iFilterColCnt = 0, uint64_t iLnStride = 0, uint64_t iColStride = 0, uint64_t iPoolIdx = POOL_MAX, uint64_t iActFuncIdx = NULL_FUNC, uint64_t iCurrLayerType = POOL, bool bIsFirstLayer = false, uint64_t iLnDilation = 0, uint64_t iColDilation = 0) : Layer(iActFuncIdx, iCurrLayerType, bIsFirstLayer), iPoolType(iPoolIdx) {InitPara(iFilterLnCnt, iFilterColCnt, iLnStride, iColStride, iLnDilation, iColDilation);}
    set<feature> ForwProp(set<feature> &setInput)
    {
        if(bFirstLayer) setLayerInput = setInput;
        else setLayerInput = std::move(setInput);
        setOutput = _CONV Pool(setLayerInput, iPoolType, true, set<feature>(), iLayerFilterLnCnt, iLayerFilterColCnt, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation);
        return Activate(setOutput);
    }
    set<feature> BackProp(set<feature> &setGrad)
    {
        return _CONV Pool(Derivative(setOutput, setGrad), PoolUpType(iPoolType), false, setLayerInput, iLayerFilterLnCnt, iLayerFilterColCnt, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation);
    }
};

class LayerBNConv : public Layer
{
protected:
    _CONV ConvBN convbnData;
    // Shift
    vect vecBeta;
    // Scale
    vect vecGamma;
    double dEpsilon = 1e-5;
    set<feature> setLayerInput;
public:
    LayerBNConv(LayerBNConv &lyrSrc) : Layer(lyrSrc)
    {
        setLayerInput = lyrSrc.setLayerInput;
        convbnData = lyrSrc.convbnData;
        vecBeta = lyrSrc.vecBeta;
        vecGamma = lyrSrc.vecGamma;
        dEpsilon = lyrSrc.dEpsilon;
    }
    LayerBNConv(LayerBNConv &&lyrSrc) : Layer(lyrSrc)
    {
        setLayerInput = std::move(lyrSrc.setLayerInput);
        convbnData = std::move(lyrSrc.convbnData);
        vecBeta = std::move(lyrSrc.vecBeta);
        vecGamma = std::move(lyrSrc.vecGamma);
        dEpsilon = lyrSrc.dEpsilon;
    }
    void operator=(LayerBNConv &lyrSrc) {new(this)LayerBNConv(lyrSrc);}
    void operator=(LayerBNConv &&lyrSrc) {new(this)LayerBNConv(std::move(lyrSrc));}
    LayerBNConv(uint64_t iChannCnt = 1, uint64_t iActFuncIdx = NULL_FUNC, double dShift = 0, double dScale = 1, uint64_t iCurrLayerType = BN_CONV, bool bIsFirstLayer = false, double dBNDominator = 1e-5) : Layer(iActFuncIdx, iCurrLayerType, bIsFirstLayer), dEpsilon(dBNDominator)
    {
        vecBeta = _CONV BNInitScaleShift(iChannCnt, dShift);
        vecGamma = _CONV BNInitScaleShift(iChannCnt, dScale);
    }
    set<feature> ForwProp(set<feature> &setInput)
    {
        if(bFirstLayer) setLayerInput = setInput;
        else setLayerInput = std::move(setInput);
        convbnData = _CONV BNTrain(setLayerInput, vecBeta, vecGamma, true, dEpsilon);
        return Activate(convbnData.setY);
    }
    set<feature> BackProp(set<feature> &setGrad, double dLearnRate)
    {
        setGrad = Derivative(convbnData.setY, setGrad);
        auto setGradBack = _CONV BNGradLossToInput(convbnData, setLayerInput, setGrad, vecGamma, dEpsilon);
        vecBeta = _CONV BNUpdateScaleShiftBat(vecBeta, _CONV BNGradLossToShift(setGrad), dLearnRate);
        vecGamma = _CONV BNUpdateScaleShiftBat(vecGamma, _CONV BNGradLossToScale(setGrad, convbnData), dLearnRate);
        return setGradBack;
    }
    set<feature> Deduce(uint64_t iMiniBatchSize) {return _CONV BNDeduce(setLayerInput, vecBeta, vecGamma, iMiniBatchSize, dEpsilon);}
};

class LayerBNConvAda : public LayerBNConv
{
protected:
    _ADA AdaDeltaVect advBeta;
    _ADA AdaDeltaVect advGamma;
public:
    LayerBNConvAda(LayerBNConvAda &lyrSrc) : LayerBNConv(lyrSrc)
    {
        advBeta = lyrSrc.advBeta;
        advGamma = lyrSrc.advGamma;
    }
    LayerBNConvAda(LayerBNConvAda &&lyrSrc) : LayerBNConv(std::move(lyrSrc))
    {
        advBeta = std::move(lyrSrc.advBeta);
        advGamma = std::move(lyrSrc.advGamma);
    }
    void operator=(LayerBNConvAda &lyrSrc) {new(this)LayerBNConvAda(lyrSrc);}
    void operator=(LayerBNConvAda &&lyrSrc) {new(this)LayerBNConvAda(std::move(lyrSrc));}
    LayerBNConvAda(uint64_t iChannCnt = 1, uint64_t iActFuncIdx = NULL_FUNC, double dShift = 0, double dScale = 1, uint64_t iCurrLayerType = BN_CONV_ADA, bool bIsFirstLayer = false, double dDecayController = 0.95, double dAdaDominator = 1e-3, double dBNDominator = 1e-10) : LayerBNConv(iChannCnt, iActFuncIdx, dShift, dScale, iCurrLayerType, bIsFirstLayer, dBNDominator)
    {
        advBeta = _ADA AdaDeltaVect(iChannCnt, 1, dDecayController, dAdaDominator);
        advGamma = _ADA AdaDeltaVect(iChannCnt, 1, dDecayController, dAdaDominator);
    }
    set<feature> BackProp(set<feature> &setGrad)
    {
        setGrad = Derivative(convbnData.setY, setGrad);
        auto setGradBack = _CONV BNGradLossToInput(convbnData, setLayerInput, setGrad, vecGamma, dEpsilon);
        vecBeta = _CONV BNAdaDeltaUpdateScaleShift(vecBeta, _CONV BNGradLossToShift(setGrad), advBeta);
        vecGamma = _CONV BNAdaDeltaUpdateScaleShift(vecGamma, _CONV BNGradLossToScale(setGrad, convbnData), advGamma);
        return setGradBack;
    }
};

class LayerTransToVect : public Layer
{
protected:
    uint64_t iLayerInputLnCnt = 0;
    uint64_t iLayerInputColCnt = 0;
public:
    set<vect> ForwProp(set<feature> &setInput)
    {
        iLayerInputLnCnt = setInput[ZERO_IDX][ZERO_IDX].LN_CNT;
        iLayerInputColCnt = setInput[ZERO_IDX][ZERO_IDX].COL_CNT;
        return _FC FeatureTransform(setInput);
    }
    set<feature> BackProp(set<vect> &setGrad) {return _FC FeatureTransform(setGrad, iLayerInputLnCnt, iLayerInputColCnt);}
    LayerTransToVect(uint64_t iActFuncIdx = NULL_FUNC, uint64_t iCurrLayerType = TRANS_TO_VECT, bool isFirstLayer = false) : Layer(iActFuncIdx, iCurrLayerType, isFirstLayer) {}
    LayerTransToVect(LayerTransToVect &lyrSrc) : Layer(lyrSrc)
    {
        iLayerInputLnCnt = lyrSrc.iLayerInputLnCnt;
        iLayerInputColCnt = lyrSrc.iLayerInputColCnt;
    }
    void operator=(LayerTransToVect &lyrSrc) {new(this)LayerTransToVect(lyrSrc);}
};

class LayerTransToFeat : public Layer
{
protected:
    uint64_t iLayerChannLnCnt = 0;
    uint64_t iLayerChannColCnt = 0;
public:
    LayerTransToFeat(uint64_t iChannLnCnt, uint64_t iChannColCnt, uint64_t iActFuncIdx = NULL_FUNC, uint64_t iCurrLayerType = TRANS_TO_FEAT, bool isFirstLayer = false) : Layer(iActFuncIdx, iCurrLayerType, isFirstLayer), iLayerChannLnCnt(iChannLnCnt), iLayerChannColCnt(iChannColCnt) {}
    set<feature> ForwProp(set<vect> &setInput) {return _FC FeatureTransform(setInput, iLayerChannLnCnt, iLayerChannColCnt);}
    set<vect> BackProp(set<feature> &setGrad) {return _FC FeatureTransform(setGrad);}
    LayerTransToFeat(LayerTransToFeat &lyrSrc) : Layer(lyrSrc)
    {
        iLayerChannLnCnt = lyrSrc.iLayerChannLnCnt;
        iLayerChannColCnt = lyrSrc.iLayerChannColCnt;
    }
    void operator=(LayerTransToFeat &lyrSrc) {new(this)LayerTransToFeat(lyrSrc);}
};

LAYER_END