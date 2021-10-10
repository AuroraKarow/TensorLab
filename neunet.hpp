NEUNET_BEGIN

template<typename CT, typename PT> const std::shared_ptr<CT> ChildInstancePtrFromParentPtr(std::shared_ptr<PT> &pParentInstance) {return std::dynamic_pointer_cast<CT>(pParentInstance);}

struct Layer
{
    uint64_t iActFuncType = SIGMOID;
    virtual void setActFunc(uint64_t iActFuncIdx) { iActFuncType = iActFuncIdx;}
    Layer() {}
    Layer(uint64_t iActFuncIdx) : iActFuncType(iActFuncIdx) {}
};

class LayerFC : public Layer
{
protected:
    vect vecWeight;
public:
    LayerFC(uint64_t iActFuncIdx = SIGMOID) : Layer(iActFuncIdx) {}
    bool InitLayerWeight(uint64_t iInputLnCnt, uint64_t iOutputLnCnt, double dRandBoundryFirst = 0.0, double dRandBoundrySecond = 0.0, double dAcc = 1e-05)
    {
        if(iInputLnCnt && iOutputLnCnt)
        {
            vecWeight = fc::InitWeight(iInputLnCnt, iOutputLnCnt, dRandBoundryFirst, dRandBoundrySecond, dAcc);
            return true;
        }
        else return false;
    }
    LayerFC(uint64_t iInputLnCnt, uint64_t iOutputLnCnt, uint64_t iActFuncIdx = SIGMOID, double dRandBoundryFirst = 0.0, double dRandBoundrySecond = 0.0, double dAcc = 1e-05) : Layer(iActFuncIdx) {InitLayerWeight(iInputLnCnt, iOutputLnCnt, dRandBoundryFirst, dRandBoundrySecond, dAcc);}
    set<vect> ForwProp(set<vect> &setInput) {return fc::Output(setInput, vecWeight);}
    set<vect> BackProp(set<vect> &setInput, set<vect> &setGrad, double dLearnRate)
    {
        auto setGradBack = fc::GradLossToInput(setGrad, vecWeight);
        vecWeight -= dLearnRate * fc::GradLossToWeight(setGrad, setInput);
        return setGradBack;
    }
};

class LayerFCAda : public LayerFC
{
protected:
    ada::AdaDeltaVect advLayerDelta;
public:
    LayerFCAda(uint64_t iActFuncIdx = SIGMOID) : LayerFC(iActFuncIdx) {}
    LayerFCAda(uint64_t iInputLnCnt, uint64_t iOutputLnCnt, uint64_t iActFuncIdx = SIGMOID, double dRandBoundryFirst = 0.0, double dRandBoundrySecond = 0.0, double dAcc = 1e-05) : LayerFC(iActFuncIdx, iInputLnCnt, iOutputLnCnt, dRandBoundryFirst, dRandBoundrySecond, dAcc) {}
    set<vect> BackProp(set<vect> &setInput, set<vect> &setGrad)
    {
        auto setGradBack = fc::GradLossToInput(setGrad, vecWeight);
        vecWeight = fc::AdaDeltaUpdateWeight(vecWeight, fc::GradLossToWeight(setGrad, setInput), advLayerDelta);
        return setGradBack;
    }
};

class LayerBNFC : public Layer
{
protected:
    fc::FCBN fcbnData;
    // Shift
    double dBeta = 0;
    // Scale
    double dGamma = 1;
    double dEpsilon = 1e-10;
public:
    LayerBNFC(double dShift = 0, double dScale = 1, double dBNDominator = 1e-10) : dBeta(dShift), dGamma(dScale), dEpsilon(dBNDominator) {}
    set<vect> ForwProp(set<vect> &setInput)
    {
        fcbnData = fc::BNTrain(setInput, dBeta, dGamma, true, dEpsilon);
        return fcbnData.setY;
    }
    set<vect> BackProp(set<vect> &setInput, set<vect> &setGrad, double dLearnRate)
    {
        auto setGradBack = fc::BNGradLossToInput(fcbnData, setInput, setGrad, dGamma, dEpsilon);
        dGamma = fc::BNUpdateScaleShift(dGamma, fc::BNGradLossToScale(setGrad, fcbnData), dLearnRate);
        dBeta = fc::BNUpdateScaleShift(dBeta, fc::BNGradLossToShift(setGrad), dLearnRate);
        return setGradBack;
    }
    set<vect> Deduce(set<vect> &setInput, uint64_t iMiniBatchSize) {return fc::BNDeduce(setInput, dBeta, dGamma, iMiniBatchSize, dEpsilon);}
};

class LayerBNFCAda : public LayerBNFC
{
protected: 
    ada::AdaDeltaVal advBeta;
    ada::AdaDeltaVal advGamma;
public:
    LayerBNFCAda(double dDecayController = 0.95, double dAdaDominator = 1e-3, double dShift = 0, double dScale = 1, double dBNDominator = 1e-10) : LayerBNFC(dShift, dScale, dBNDominator)
    {
        advBeta = ada::AdaDeltaVal(0, 0, dDecayController, dAdaDominator);
        advGamma = ada::AdaDeltaVal(0, 0, dDecayController, dAdaDominator);
    }
    set<vect> BackProp(set<vect> &setInput, set<vect> &setGrad)
    {
        auto setGradBack = fc::BNGradLossToInput(fcbnData, setInput, setGrad, dGamma, dEpsilon);
        dGamma = fc::AdaDeltaUpdateScaleShift(dGamma, fc::BNGradLossToScale(setGrad, fcbnData), advGamma);
        dBeta = fc::AdaDeltaUpdateScaleShift(dBeta, fc::BNGradLossToShift(setGrad), advBeta);
        return setGradBack;
    }
};

class LayerConv : public Layer
{
protected:
    tensor tenKernel;
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
public:
    LayerConv(uint64_t iActFuncIdx = RELU) : Layer(iActFuncIdx) {}
    bool InitLayerKernel(uint64_t iAmt, uint64_t iChannCnt, uint64_t iLnCnt, uint64_t iColCnt, double dRandBoundryFirst = 0, double dRandBoundrySecond = 0, double dAcc = 1e-5)
    {
        if(iAmt && iChannCnt && iLnCnt && iColCnt)
        {
            tenKernel = conv::InitKernel(iAmt, iChannCnt, iLnCnt, iColCnt, dRandBoundryFirst, dRandBoundrySecond, dAcc);
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

    LayerConv(uint64_t iKernelAmt, uint64_t iKernelChannCnt, uint64_t iKernelLnCnt, uint64_t iKernelColCnt, uint64_t iLnStride, uint64_t iColStride, uint64_t iActFuncIdx = RELU, double dRandBoundryFirst = 0, double dRandBoundrySecond = 0, double dAcc = 1e-5, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0) : Layer(iActFuncIdx)
    {
        InitLayerKernel(iKernelAmt, iKernelChannCnt, iKernelLnCnt, iKernelColCnt, dRandBoundryFirst, dRandBoundrySecond, dAcc);
        InitPara(iLnStride, iColStride, iLnDilation, iColDilation, iInputPadTop, iInputPadRight, iInputPadBottom, iInputPadLeft, iLnDistance, iColDistance);
    }

    set<feature> ForwProp(set<feature> &setInput) {return MRG_CHANN(conv::Conv(setInput, tenKernel, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance));}

    set<feature> BackProp(set<feature> &setInput, set<feature> &setGrad, double dLearnRate) 
    {
        auto setGradBack = conv::GradLossToInput(setGrad, tenKernel, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance);
        tenKernel = conv::UpdateKernel(tenKernel, conv::GradLossToKernel(setGrad, setInput, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance), dLearnRate);
        return setGradBack;
    }
};

class LayerConvAda : public LayerConv
{
protected:
    ada::ten_ada<ada::AdaDeltaVect> advLayerDelta;
public:
    LayerConvAda(uint64_t iActFuncIdx = RELU) : LayerConv(iActFuncIdx) {}
    LayerConvAda(uint64_t iKernelAmt, uint64_t iKernelChannCnt, uint64_t iKernelLnCnt, uint64_t iKernelColCnt, uint64_t iLnStride, uint64_t iColStride, uint64_t iActFuncIdx = RELU, double dRandBoundryFirst = 0, double dRandBoundrySecond = 0, double dAcc = 1e-5, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0) : LayerConv(iKernelAmt, iKernelChannCnt, iKernelLnCnt, iKernelColCnt, iLnStride, iColStride, iActFuncIdx, dRandBoundryFirst, dRandBoundrySecond, dAcc, iLnDilation, iColDilation, iInputPadTop, iInputPadRight, iInputPadBottom, iInputPadLeft, iLnDistance, iColDistance) {}
    set<feature> BackProp(set<feature> &setInput, set<feature> &setGrad)
    {
        auto setGradBack = conv::GradLossToInput(setGrad, tenKernel, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance);
        tenKernel = conv::AdaDeltaUpdateKernel(tenKernel, conv::GradLossToKernel(setGrad, setInput, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance), advLayerDelta);
        return setGradBack;
    }
};

class LayerPool : public Layer
{
protected:
    uint64_t iPoolType = POOL_MAX;
    uint64_t iLayerFilterLnCnt = 0;
    uint64_t iLayerFilterColCnt = 0;
    uint64_t iLayerLnStride = 0;
    uint64_t iLayerColStride = 0;
    uint64_t iLayerLnDilation = 0;
    uint64_t iLayerColDilation = 0;
    uint64_t PoolUpType()
    {
        switch (iPoolType)
        {
        case POOL_AVG: return POOL_UP_AVG;
        case POOL_GAG: return POOL_UP_GAG;
        case POOL_MAX: return POOL_UP_MAX;
        default: return (uint64_t)NAN;
        }
    }
public:
    LayerPool(uint64_t iPoolIdx = POOL_MAX) : iPoolType(iPoolIdx) {}
    void InitPara(uint64_t iFilterLnCnt = 0, uint64_t iFilterColCnt = 0, uint64_t iLnStride = 0, uint64_t iColStride = 0, uint64_t iLnDilation = 0, uint64_t iColDilation = 0)
    {
        iLayerFilterLnCnt = iFilterLnCnt; iLayerFilterColCnt = iFilterColCnt;
        iLayerLnStride = iLnStride; iLayerColStride = iColStride;
        iLayerLnDilation = iLnDilation; iLayerColDilation = iColDilation;
    }
    LayerPool(uint64_t iFilterLnCnt = 0, uint64_t iFilterColCnt = 0, uint64_t iLnStride = 0, uint64_t iColStride = 0, uint64_t iPoolIdx = POOL_MAX, uint64_t iLnDilation = 0, uint64_t iColDilation = 0) : iPoolType(iPoolIdx) {InitPara(iFilterLnCnt, iFilterColCnt, iLnStride, iColStride, iLnDilation, iColDilation);}
    set<feature> ForwProp(set<feature> &setInput) {return conv::Pool(setInput, iPoolType, true, set<feature>(), iLayerFilterLnCnt, iLayerFilterColCnt, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation);}
    set<feature> BackProp(set<feature> &setGrad, set<feature> &setInput = set<feature>()) {return conv::Pool(setGrad, PoolUpType(), false, setInput, iLayerFilterLnCnt, iLayerFilterColCnt, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation);}
};

class LayerBNConv : public Layer
{
protected:
    conv::ConvBN convbnData;
    // Shift
    vect vecBeta;
    // Scale
    vect vecGamma;
    double dEpsilon = 1e-5;
public:
    LayerBNConv(double iChannCnt, double dShift = 0, double dScale = 1, double dBNDominator = 1e-5) : dEpsilon(dBNDominator)
    {
        vecBeta = conv::BNInitScaleShift(iChannCnt, dShift);
        vecGamma = conv::BNInitScaleShift(iChannCnt, dScale);
    }
    
};

NEUNET_END