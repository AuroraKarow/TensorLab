LAYER_BEGIN

class Layer
{
protected:
    uint64_t iActFuncType = SIGMOID;
    bool bFirstLayer = false;
    template <typename inputT> struct InputData{inputT vecLayerInput;};
    uint64_t iLayerType = FC;
public:
    virtual void setActFunc(uint64_t iActFuncIdx) { iActFuncType = iActFuncIdx;}
    Layer(uint64_t iActFuncIdx = SIGMOID, bool bIsFirstLayer = false) : iActFuncType(iActFuncIdx), bFirstLayer(bIsFirstLayer) {}
    void operator=(Layer &lyrSrc) {iActFuncType = lyrSrc.iActFuncType;}
    void operator=(Layer &&lyrSrc) {iActFuncType = std::move(lyrSrc.iActFuncType);}
    Layer(Layer &lyrSrc) {*this = lyrSrc;}
    Layer(Layer &&lyrSrc) {*this = std::move(lyrSrc);}
    void ForwProp() {}
    void BackProp() {}
    void InitPara() {}
};

class LayerFC : public Layer
{
protected:
    vect vecLayerWeight;
    set<vect> vecLayerInput;
public:
    LayerFC(uint64_t iActFuncIdx = SIGMOID) : Layer(iActFuncIdx) {}
    bool InitLayerWeight(uint64_t iInputLnCnt, uint64_t iOutputLnCnt, double dRandBoundryFirst = 0.0, double dRandBoundrySecond = 0.0, double dAcc = 1e-05)
    {
        if(iInputLnCnt && iOutputLnCnt)
        {
            vecLayerWeight = _FC InitWeight(iInputLnCnt, iOutputLnCnt, dRandBoundryFirst, dRandBoundrySecond, dAcc);
            return true;
        }
        else return false;
    }
    LayerFC(uint64_t iInputLnCnt, uint64_t iOutputLnCnt, uint64_t iActFuncIdx = SIGMOID, bool bIsFirstLayer = false, double dRandBoundryFirst = 0.0, double dRandBoundrySecond = 0.0, double dAcc = 1e-05) : Layer(iActFuncIdx, bIsFirstLayer) {InitLayerWeight(iInputLnCnt, iOutputLnCnt, dRandBoundryFirst, dRandBoundrySecond, dAcc);}
    set<vect> ForwProp(set<vect> &setInput)
    {
        return _FC Output(setInput, vecLayerWeight);
        if(bFirstLayer)vecLayerInput = setInput;
        else vecLayerInput = std::move(setInput);
    }
    set<vect> BackProp(set<vect> &setGrad, double dLearnRate)
    {
        auto setGradBack = _FC GradLossToInput(setGrad, vecLayerWeight);
        vecLayerWeight -= dLearnRate * _FC GradLossToWeight(setGrad, vecLayerInput);
        return setGradBack;
    }
    LayerFC(LayerFC &lyrSrc) {*this = lyrSrc;}
    LayerFC(LayerFC &&lyrSrc) {*this = std::move(lyrSrc);}
    void operator=(LayerFC &lyrSrc)
    {
        Layer::operator=(lyrSrc);
        vecLayerWeight = lyrSrc.vecLayerWeight;
        vecLayerInput = lyrSrc.vecLayerInput;
    }
    void operator=(LayerFC &&lyrSrc)
    {
        Layer::operator=(std::move(lyrSrc));
        vecLayerWeight = std::move(lyrSrc.vecLayerWeight);
        vecLayerInput = std::move(lyrSrc.vecLayerInput);
    }
};

class LayerFCAda : public LayerFC
{
protected:
    _ADA AdaDeltaVect advLayerDelta;
public:
    LayerFCAda(uint64_t iActFuncIdx = SIGMOID) : LayerFC(iActFuncIdx) {}
    LayerFCAda(uint64_t iInputLnCnt, uint64_t iOutputLnCnt, uint64_t iActFuncIdx = SIGMOID, bool bIsFirstLayer = false, double dRandBoundryFirst = 0.0, double dRandBoundrySecond = 0.0, double dAcc = 1e-05) : LayerFC(iInputLnCnt, iOutputLnCnt, iActFuncIdx, bIsFirstLayer, dRandBoundryFirst, dRandBoundrySecond, dAcc) {}
    set<vect> BackProp(set<vect> &setInput, set<vect> &setGrad)
    {
        auto setGradBack = _FC GradLossToInput(setGrad, vecLayerWeight);
        vecLayerWeight = _FC AdaDeltaUpdateWeight(vecLayerWeight, _FC GradLossToWeight(setGrad, setInput), advLayerDelta);
        return setGradBack;
    }
    LayerFCAda(LayerFCAda &lyrSrc) {*this = lyrSrc;}
    LayerFCAda(LayerFCAda &&lyrSrc) {*this = std::move(lyrSrc);}
    void operator=(LayerFCAda &lyrSrc)
    {
        LayerFC::operator=(lyrSrc);
        advLayerDelta = lyrSrc.advLayerDelta;
    }
    void operator=(LayerFCAda &&lyrSrc)
    {
        LayerFC::operator=(std::move(lyrSrc));
        advLayerDelta = std::move(lyrSrc.advLayerDelta);
    }
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
    LayerBNFC(bool bIsFirstLayer = false, double dShift = 0, double dScale = 1, double dBNDominator = 1e-10) : Layer(0, bIsFirstLayer), dBeta(dShift), dGamma(dScale), dEpsilon(dBNDominator) {}
    set<vect> ForwProp(set<vect> &setInput)
    {
        if(bFirstLayer) setLayerInput = setInput;
        else setLayerInput = std::move(setInput);
        fcbnData = _FC BNTrain(setInput, dBeta, dGamma, true, dEpsilon);
        return fcbnData.setY;
    }
    set<vect> BackProp(set<vect> &setGrad, double dLearnRate)
    {
        auto setGradBack = _FC BNGradLossToInput(fcbnData, setLayerInput, setGrad, dGamma, dEpsilon);
        dGamma = _FC BNUpdateScaleShift(dGamma, _FC BNGradLossToScale(setGrad, fcbnData), dLearnRate);
        dBeta = _FC BNUpdateScaleShift(dBeta, _FC BNGradLossToShift(setGrad), dLearnRate);
        return setGradBack;
    }
    set<vect> Deduce(uint64_t iMiniBatchSize) {return _FC BNDeduce(setLayerInput, dBeta, dGamma, iMiniBatchSize, dEpsilon);}
    LayerBNFC(LayerBNFC &lyrSrc) {*this = lyrSrc;}
    LayerBNFC(LayerBNFC &&lyrSrc) {*this = std::move(lyrSrc);}
    void operator=(LayerBNFC &lyrSrc)
    {
        Layer::operator=(lyrSrc);
        fcbnData = lyrSrc.fcbnData;
        ValueAssign(lyrSrc);
        setLayerInput = lyrSrc.setLayerInput;
    }
    void operator=(LayerBNFC &&lyrSrc)
    {
        Layer::operator=(std::move(lyrSrc));
        fcbnData = std::move(lyrSrc.fcbnData);
        ValueAssign(lyrSrc);
        setLayerInput = std::move(lyrSrc.setLayerInput);
    }
};

class LayerBNFCAda : public LayerBNFC
{
protected: 
    _ADA AdaDeltaVal advBeta;
    _ADA AdaDeltaVal advGamma;
public:
    LayerBNFCAda(bool bIsFirstLayer = false, double dDecayController = 0.95, double dAdaDominator = 1e-3, double dShift = 0, double dScale = 1, double dBNDominator = 1e-10) : LayerBNFC(bIsFirstLayer, dShift, dScale, dBNDominator)
    {
        advBeta = _ADA AdaDeltaVal(0, 0, dDecayController, dAdaDominator);
        advGamma = _ADA AdaDeltaVal(0, 0, dDecayController, dAdaDominator);
    }
    set<vect> BackProp(set<vect> &setGrad)
    {
        auto setGradBack = _FC BNGradLossToInput(fcbnData, setLayerInput, setGrad, dGamma, dEpsilon);
        dGamma = _FC BNAdaDeltaUpdateScaleShift(dGamma, _FC BNGradLossToScale(setGrad, fcbnData), advGamma);
        dBeta = _FC BNAdaDeltaUpdateScaleShift(dBeta, _FC BNGradLossToShift(setGrad), advBeta);
        return setGradBack;
    }
    LayerBNFCAda(LayerBNFCAda &lyrSrc) {*this = lyrSrc;}
    void operator=(LayerBNFCAda &lyrSrc)
    {
        LayerBNFC::operator=(lyrSrc);
        advBeta = lyrSrc.advBeta;
        advGamma = lyrSrc.advGamma;
    }
};

class LayerConv : public Layer
{
protected:
    tensor tenKernel;
    set<feature> setLayerInput;
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
    LayerConv(LayerConv &lyrSrc) {*this = lyrSrc;}
    LayerConv(LayerConv &&lyrSrc) {*this = std::move(lyrSrc);}
    void operator=(LayerConv &lyrSrc)
    {
        Layer::operator=(lyrSrc);
        tenKernel = lyrSrc.tenKernel;
        setLayerInput = lyrSrc.setLayerInput;
        ValueAssign(lyrSrc);
    }
    void operator=(LayerConv &&lyrSrc)
    {
        Layer::operator=(std::move(lyrSrc));
        tenKernel = std::move(lyrSrc.tenKernel);
        ValueAssign(lyrSrc);
        setLayerInput = std::move(lyrSrc.setLayerInput);
    }
    LayerConv(uint64_t iActFuncIdx = RELU) : Layer(iActFuncIdx) {}
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
    LayerConv(uint64_t iKernelAmt, uint64_t iKernelChannCnt, uint64_t iKernelLnCnt, uint64_t iKernelColCnt, uint64_t iLnStride, uint64_t iColStride, uint64_t iActFuncIdx = RELU, bool bIsFirstLayer = false, double dRandBoundryFirst = 0, double dRandBoundrySecond = 0, double dAcc = 1e-5, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0) : Layer(iActFuncIdx, bIsFirstLayer)
    {
        InitLayerKernel(iKernelAmt, iKernelChannCnt, iKernelLnCnt, iKernelColCnt, dRandBoundryFirst, dRandBoundrySecond, dAcc);
        InitPara(iLnStride, iColStride, iLnDilation, iColDilation, iInputPadTop, iInputPadRight, iInputPadBottom, iInputPadLeft, iLnDistance, iColDistance);
    }
    set<feature> ForwProp(set<feature> &setInput)
    {
        if(bFirstLayer) setLayerInput = setInput;
        else setLayerInput = std::move(setInput);
        return MRG_CHANN(_CONV Conv(setInput, tenKernel, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance));
    }
    set<feature> BackProp(set<feature> &setGrad, double dLearnRate) 
    {
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
    LayerConvAda(LayerConvAda &lyrSrc) {*this = lyrSrc;}
    LayerConvAda(LayerConvAda &&lyrSrc) {*this = std::move(lyrSrc);}
    void operator=(LayerConvAda &lyrSrc)
    {
        LayerConv::operator=(lyrSrc);
        advLayerDelta = lyrSrc.advLayerDelta;
    }
    void operator=(LayerConvAda &&lyrSrc)
    {
        LayerConv::operator=(std::move(lyrSrc));
        advLayerDelta = std::move(lyrSrc.advLayerDelta);
    }
    LayerConvAda(uint64_t iActFuncIdx = RELU) : LayerConv(iActFuncIdx) {}
    LayerConvAda(uint64_t iKernelAmt, uint64_t iKernelChannCnt, uint64_t iKernelLnCnt, uint64_t iKernelColCnt, uint64_t iLnStride, uint64_t iColStride, uint64_t iActFuncIdx = RELU, bool bIsFirstLayer = false, double dRandBoundryFirst = 0, double dRandBoundrySecond = 0, double dAcc = 1e-5, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0) : LayerConv(iKernelAmt, iKernelChannCnt, iKernelLnCnt, iKernelColCnt, iLnStride, iColStride, iActFuncIdx, bIsFirstLayer, dRandBoundryFirst, dRandBoundrySecond, dAcc, iLnDilation, iColDilation, iInputPadTop, iInputPadRight, iInputPadBottom, iInputPadLeft, iLnDistance, iColDistance) {}
    set<feature> BackProp(set<feature> &setGrad)
    {
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
    LayerPool(LayerPool &lyrSrc) {*this = lyrSrc;}
    LayerPool(LayerPool &&lyrSrc) {*this = std::move(lyrSrc);}
    void operator=(LayerPool &lyrSrc)
    {
        Layer::operator=(lyrSrc);
        setLayerInput = lyrSrc.setLayerInput;
        ValueAssign(lyrSrc);
    }
    void operator=(LayerPool &&lyrSrc)
    {
        Layer::operator=(std::move(lyrSrc));
        ValueAssign(lyrSrc);
        setLayerInput = std::move(lyrSrc.setLayerInput);
    }
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
    LayerPool(uint64_t iPoolIdx = POOL_MAX, bool bIsFirstLayer = false) : Layer(0, bIsFirstLayer), iPoolType(iPoolIdx) {}
    void InitPara(uint64_t iFilterLnCnt = 0, uint64_t iFilterColCnt = 0, uint64_t iLnStride = 0, uint64_t iColStride = 0, uint64_t iLnDilation = 0, uint64_t iColDilation = 0)
    {
        iLayerFilterLnCnt = iFilterLnCnt; iLayerFilterColCnt = iFilterColCnt;
        iLayerLnStride = iLnStride; iLayerColStride = iColStride;
        iLayerLnDilation = iLnDilation; iLayerColDilation = iColDilation;
    }
    LayerPool(uint64_t iFilterLnCnt = 0, uint64_t iFilterColCnt = 0, uint64_t iLnStride = 0, bool bIsFirstLayer = false, uint64_t iColStride = 0, uint64_t iPoolIdx = POOL_MAX, uint64_t iLnDilation = 0, uint64_t iColDilation = 0) : Layer(0, bIsFirstLayer), iPoolType(iPoolIdx) {InitPara(iFilterLnCnt, iFilterColCnt, iLnStride, iColStride, iLnDilation, iColDilation);}
    set<feature> ForwProp(set<feature> &setInput)
    {
        if(bFirstLayer) setLayerInput = setInput;
        else setLayerInput = std::move(setInput);
        return _CONV Pool(setInput, iPoolType, true, set<feature>(), iLayerFilterLnCnt, iLayerFilterColCnt, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation);
    }
    set<feature> BackProp(set<feature> &setGrad) {return _CONV Pool(setGrad, PoolUpType(iPoolType), false, setLayerInput, iLayerFilterLnCnt, iLayerFilterColCnt, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation);}
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
    LayerBNConv(LayerBNConv &lyrSrc) {*this = lyrSrc;}
    LayerBNConv(LayerBNConv &&lyrSrc) {*this = std::move(lyrSrc);}
    void operator=(LayerBNConv &lyrSrc)
    {
        Layer::operator=(lyrSrc);
        setLayerInput = lyrSrc.setLayerInput;
        convbnData = lyrSrc.convbnData;
        vecBeta = lyrSrc.vecBeta;
        vecGamma = lyrSrc.vecGamma;
        dEpsilon = lyrSrc.dEpsilon;
    }
    void operator=(LayerBNConv &&lyrSrc)
    {
        Layer::operator=(std::move(lyrSrc));
        setLayerInput = std::move(lyrSrc.setLayerInput);
        convbnData = std::move(lyrSrc.convbnData);
        vecBeta = std::move(lyrSrc.vecBeta);
        vecGamma = std::move(lyrSrc.vecGamma);
        dEpsilon = lyrSrc.dEpsilon;
    }
    LayerBNConv(bool bIsFirstLayer = false, uint64_t iChannCnt = 1, double dShift = 0, double dScale = 1, double dBNDominator = 1e-5) : Layer(0, bIsFirstLayer), dEpsilon(dBNDominator)
    {
        vecBeta = _CONV BNInitScaleShift(iChannCnt, dShift);
        vecGamma = _CONV BNInitScaleShift(iChannCnt, dScale);
    }
    set<feature> ForwProp(set<feature> &setInput)
    {
        if(bFirstLayer) setLayerInput = setInput;
        else setLayerInput = std::move(setInput);
        convbnData = _CONV BNTrain(setInput, vecBeta, vecGamma, true, dEpsilon);
        return convbnData.setY;
    }
    set<feature> BackProp(set<feature> &setGrad, double dLearnRate)
    {
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
    LayerBNConvAda(LayerBNConvAda &lyrSrc) {*this = lyrSrc;}
    LayerBNConvAda(LayerBNConvAda &&lyrSrc) {*this = std::move(lyrSrc);}
    void operator=(LayerBNConvAda &lyrSrc)
    {
        LayerBNConv::operator=(lyrSrc);
        advBeta = lyrSrc.advBeta;
        advGamma = lyrSrc.advGamma;
    }
    void operator=(LayerBNConvAda &&lyrSrc)
    {
        LayerBNConv::operator=(std::move(lyrSrc));
        advBeta = std::move(lyrSrc.advBeta);
        advGamma = std::move(lyrSrc.advGamma);
    }
    LayerBNConvAda(bool bIsFirstLayer = false, uint64_t iChannCnt = 1, double dDecayController = 0.95, double dAdaDominator = 1e-3, double dShift = 0, double dScale = 1, double dBNDominator = 1e-10) : LayerBNConv(bIsFirstLayer, iChannCnt, dShift, dScale, dBNDominator)
    {
        advBeta = _ADA AdaDeltaVect(iChannCnt, 1, dDecayController, dAdaDominator);
        advGamma = _ADA AdaDeltaVect(iChannCnt, 1, dDecayController, dAdaDominator);
    }
    set<feature> BackProp(set<feature> &setGrad)
    {
        auto setGradBack = _CONV BNGradLossToInput(convbnData, setLayerInput, setGrad, vecGamma, dEpsilon);
        vecBeta = _CONV BNAdaDeltaUpdateScaleShift(vecBeta, _CONV BNGradLossToShift(setGrad), advBeta);
        vecGamma = _CONV BNAdaDeltaUpdateScaleShift(vecGamma, _CONV BNGradLossToScale(setGrad, convbnData), advGamma);
        return setGradBack;
    }
};

class LayerTransFeature : public Layer
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
    LayerTransFeature(bool isFirstLayer = false) : Layer(0, isFirstLayer) {}
    LayerTransFeature(LayerTransFeature &lyrSrc) {*this = lyrSrc;}
    void operator=(LayerTransFeature &lyrSrc)
    {
        Layer::operator=(lyrSrc);
        iLayerInputLnCnt = lyrSrc.iLayerInputLnCnt;
        iLayerInputColCnt = lyrSrc.iLayerInputColCnt;
    }
};

class LayerTransVect : public Layer
{
protected:
    uint64_t iLayerChannLnCnt = 0;
    uint64_t iLayerChannColCnt = 0;
public:
    LayerTransVect(uint64_t iChannLnCnt, uint64_t iChannColCnt, bool isFirstLayer = false) : Layer(0, isFirstLayer), iLayerChannLnCnt(iChannLnCnt), iLayerChannColCnt(iChannColCnt) {}
    set<feature> ForwProp(set<vect> &setInput) {return _FC FeatureTransform(setInput, iLayerChannLnCnt, iLayerChannColCnt);}
    set<vect> BackProp(set<feature> &setGrad) {return _FC FeatureTransform(setGrad);}
    LayerTransVect(LayerTransVect &lyrSrc) {*this = lyrSrc;}
    void operator=(LayerTransVect &lyrSrc)
    {
        Layer::operator=(lyrSrc);
        iLayerChannLnCnt = lyrSrc.iLayerChannLnCnt;
        iLayerChannColCnt = lyrSrc.iLayerChannColCnt;
    }
};

LAYER_END